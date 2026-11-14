//! One-step relational probes as circuits: the vector operations the
//! reasoner's operators and its write path are built from.
//!
//! **`relation_probe`**: inputs `fillers` (K x 2d) and `role_sel` (K x 5,
//! one-hot role per filler), and `polarity` (1 x 1). The query is the bundle
//! `q = sum_k role_k (*) filler_k` (output `q`: also the row a new statement
//! is stored as); a slot matches when it is active and
//! `sim(slot, q) >= K - margin` (`sim` = Re<a, b> / d). Outputs (1 x S):
//! - `hits`: matching slots;
//! - `hits_pol`: matching slots whose polarity equals `polarity`;
//! - `dups`: those whose stored role count also equals K (the same statement).
//!
//! **`symbol_cleanup`**: input `v` (1 x 2d) and `mode` (1 x 3 one-hot: cleanup,
//! same symbol, pronoun). Cosine of `v` with every codebook row (rows are unit
//! phasors, so cos = row . v / (sqrt(d) |v|)); the best row, and `found` when
//! its cosine reaches the mode's threshold. A zero `v` has cosine 0 everywhere.
//!
//! Blocks (units of ablation): probe `query` (role selection), `match`
//! (margin, 1/d), `filter` (metadata selectors); cleanup `cosine`
//! (1/sqrt(d), norm epsilon), `threshold` (the three thresholds).

use crate::compiler::{Builder, Compiled};
use crate::circuits::relational::{META_ACTIVE, META_POLARITY, N_ROLES};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn::*, Instr, IsaError, NormKind, ReduceFn};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeParams {
    pub dim: usize,
    pub meta_cols: usize,
    /// Metadata column holding the number of stored role-filler pairs.
    pub n_roles_col: usize,
    pub match_margin: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanupParams {
    pub dim: usize,
    pub cleanup_min: f32,
    pub symbol_same: f32,
    pub pronoun_min: f32,
}

pub const PROBE_OUTPUTS: [&str; 4] = ["q", "hits", "hits_pol", "dups"];
pub const CLEANUP_OUTPUTS: [&str; 3] = ["best", "cos", "found"];
/// `mode` columns of `symbol_cleanup`.
pub const MODE_CLEANUP: usize = 0;
pub const MODE_SAME: usize = 1;
pub const MODE_PRONOUN: usize = 2;

fn onehot(n: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

fn eq(b: &mut Builder, a: &str, x: &str, out: &str) {
    b.gate(Ge, a, x, &format!("{out}_lo"));
    b.gate(Ge, x, a, &format!("{out}_hi"));
    b.gate(Mul, &format!("{out}_lo"), &format!("{out}_hi"), out);
}

pub fn compile_probe(id: &str, p: &ProbeParams) -> Result<Compiled, IsaError> {
    if p.dim == 0 || p.meta_cols <= META_POLARITY || p.n_roles_col >= p.meta_cols {
        return Err(IsaError(format!("{id}: invalid relation_probe parameters")));
    }
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.input_dyn("fillers", None, Some(d2));
    b.input_dyn("role_sel", None, Some(N_ROLES));
    b.input("polarity", Some(1), 1);
    b.external_dyn("slots", None, Some(d2));
    b.external_dyn("meta", None, Some(p.meta_cols));
    b.external("roles", N_ROLES, d2);

    b.block("query");
    b.matmul_ab("role_sel", "roles", "role_rows");
    b.op(Instr::Bind { a: "role_rows".into(), b: "fillers".into(), out: "pairs".into(), conj: false });
    b.reduce(ReduceFn::Sum, "pairs", Axis::Cols, "q");
    b.reduce(ReduceFn::Sum, "role_sel", Axis::All, "k");

    b.block("match");
    b.scalar("margin", p.match_margin);
    b.scalar("inv_d", 1.0 / p.dim as f32);
    b.gate(Sub, "k", "margin", "need");
    b.matvec("slots", "q", "dot", false);
    b.gate(Mul, "dot", "inv_d", "sim");
    b.gate(Ge, "sim", "need", "close");

    b.block("filter");
    b.scalar("half", 0.5);
    b.constant("active_sel", 1, p.meta_cols, onehot(p.meta_cols, META_ACTIVE));
    b.constant("polarity_sel", 1, p.meta_cols, onehot(p.meta_cols, META_POLARITY));
    b.constant("n_roles_sel", 1, p.meta_cols, onehot(p.meta_cols, p.n_roles_col));
    b.matvec("meta", "active_sel", "active", false);
    b.gate(Ge, "active", "half", "is_active");
    b.gate(Mul, "close", "is_active", "hits");
    b.matvec("meta", "polarity_sel", "pol", false);
    eq(&mut b, "pol", "polarity", "pol_eq");
    b.gate(Mul, "hits", "pol_eq", "hits_pol");
    b.matvec("meta", "n_roles_sel", "n_roles", false);
    eq(&mut b, "n_roles", "k", "roles_eq");
    b.gate(Mul, "hits_pol", "roles_eq", "dups");
    Ok(b.finish(&PROBE_OUTPUTS))
}

pub fn compile_cleanup(id: &str, p: &CleanupParams) -> Result<Compiled, IsaError> {
    if p.dim == 0 {
        return Err(IsaError(format!("{id}: invalid symbol_cleanup parameters")));
    }
    let d2 = 2 * p.dim;
    let mut b = Builder::new(id);
    b.input("v", Some(1), d2);
    b.input("mode", Some(1), 3);
    b.external_dyn("codebook", None, Some(d2));

    b.block("cosine");
    b.scalar("eps", 0.0);
    b.scalar("inv_sqrt_d", 1.0 / (p.dim as f32).sqrt());
    b.op(Instr::Norm { kind: NormKind::L2, x: "v".into(), eps: "eps".into(), out: "unit".into() });
    b.matvec("codebook", "unit", "dot", false);
    b.gate(Mul, "dot", "inv_sqrt_d", "cosines");
    b.op(Instr::Topk { x: "cosines".into(), k: 1, idx: Some("best".into()), val: Some("cos".into()), mask: None });

    b.block("threshold");
    b.constant("thresholds", 3, 1, vec![p.cleanup_min, p.symbol_same, p.pronoun_min]);
    b.matmul_ab("mode", "thresholds", "min");
    b.gate(Ge, "cos", "min", "found");
    Ok(b.finish(&CLEANUP_OUTPUTS))
}
