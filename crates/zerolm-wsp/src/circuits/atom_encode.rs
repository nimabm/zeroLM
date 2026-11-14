//! The vector-symbolic atom encoder as a circuit: the bytes of a namespaced
//! name in, one unit phasor vector out. Symbols, roles, and relation atoms of
//! the reasoner are computed here (not by a host random generator), so the
//! ISA can compute every name's vector itself.
//!
//! Input `name` is 1 x L byte values: the namespace-qualified name (for
//! example `sym:cat`) at positions 1..=len, zero elsewhere; position 0 stays
//! 0 (it carries the salt). The host copies bytes and inspects none.
//!
//! Three salted copies of the row (salt byte 1, 2, 3 at position 0) are each
//! hashed whole (`hash_rows` with window n = L: exactly one window) into a
//! seeded random table of `buckets` rows. Each hashed row is normalized per
//! complex component to a unit phasor, and the three are bound (complex
//! product). Two different names collide only if all three hashes collide
//! (probability about buckets^-3), and unrelated names give independent
//! phases, so their cosine is about 0 (sd 1 / sqrt(2 dim)).
//!
//! Blocks (units of ablation):
//! - `salt`: the broadcast column of ones and the salt matrix.
//! - `select`: the one-hot row selectors.
//! - `support`: the norm epsilon (0: a zero row stays zero).

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Instr, IsaError, NormKind};

/// Hash copies bound together.
pub const COPIES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    /// Input width L in bytes (position 0 is the salt).
    pub name_bytes: usize,
    /// Rows of the seeded random table (`core.reasoner.atom_table`).
    pub buckets: usize,
    /// Complex dimension of the output vector.
    pub dim: usize,
}

pub const OUTPUTS: [&str; 1] = ["atom"];

pub fn compile(id: &str, p: &Params) -> Result<Compiled, IsaError> {
    if p.name_bytes < 4 || p.buckets == 0 || p.dim == 0 {
        return Err(IsaError(format!("{id}: invalid atom_encode parameters")));
    }
    let (l, d) = (p.name_bytes, p.dim);
    let mut b = Builder::new(id);
    b.input("name", Some(1), l);
    b.external("table", p.buckets, 2 * d);

    b.block("support");
    b.scalar("eps", 0.0);

    // Three salted copies of the name row.
    b.block("salt");
    b.constant("ones", COPIES, 1, vec![1.0; COPIES]);
    let mut salts = vec![0.0f32; COPIES * l];
    for j in 0..COPIES {
        salts[j * l] = (j + 1) as f32;
    }
    b.constant("salts", COPIES, l, salts);
    b.matmul_ab("ones", "name", "copies");
    b.gate(zerolm_engine::isa::GateFn::Add, "copies", "salts", "salted");

    // One whole-row window per copy: COPIES x 2d rows of the table.
    b.op(Instr::HashRows { bytes: "salted".into(), table: "table".into(), n: l, out: "hashed".into() });

    // Per complex component (re_k, im_k) to a unit phasor, for all copies at once.
    b.reshape("hashed", 2 * COPIES, d, "split");
    b.op(Instr::Transpose { x: "split".into(), out: "by_comp".into() });
    b.reshape("by_comp", COPIES * d, 2, "pairs");
    b.op(Instr::Norm { kind: NormKind::L2, x: "pairs".into(), eps: "eps".into(), out: "unit_pairs".into() });
    b.reshape("unit_pairs", d, 2 * COPIES, "unit_by_comp");
    b.op(Instr::Transpose { x: "unit_by_comp".into(), out: "unit_split".into() });
    b.reshape("unit_split", COPIES, 2 * d, "units");

    // Bind the copies.
    b.block("select");
    for j in 0..COPIES {
        let mut sel = vec![0.0f32; COPIES];
        sel[j] = 1.0;
        b.constant(&format!("pick{j}"), 1, COPIES, sel);
        b.matmul_ab(&format!("pick{j}"), "units", &format!("u{j}"));
    }
    b.op(Instr::Bind { a: "u0".into(), b: "u1".into(), out: "u01".into(), conj: false });
    b.op(Instr::Bind { a: "u01".into(), b: "u2".into(), out: "atom".into(), conj: false });

    Ok(b.finish(&OUTPUTS))
}
