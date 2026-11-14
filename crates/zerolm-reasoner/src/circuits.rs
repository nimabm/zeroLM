//! The compiled relational circuits (roadmap C5, C6) bound to this
//! reasoner's tensors. The host binds the used prefix of the relation slots
//! and the codebook, runs the programs, and reads their outputs.
//!
//! - `relation-graph` (C5) decodes the memory once per memory state; the
//!   result is cached session state, recomputed after any write.
//! - `relation-reach` (C6) replaces breadth-first search: reachability and
//!   shortest-proof premises as a fixed-step recurrence.

use crate::{field, Reasoner, ReasonerConfig, ROLES};
use std::collections::BTreeMap;
use zerolm_engine::binding::Hv;
use zerolm_engine::isa::{self, bundle, Bindings, Buf, Outputs, View};
use zerolm_wsp::circuits::atom_encode::Params as AtomParams;
use zerolm_wsp::circuits::relational::{Params, OPERATORS};
use zerolm_wsp::{compile, Circuit, Compiled, Spec};

pub const GRAPH_ID: &str = "relation-graph";
pub const REACH_ID: &str = "relation-reach";

fn params(cfg: &ReasonerConfig) -> Params {
    Params {
        dim: cfg.dim,
        meta_cols: field::COUNT,
        max_depth: cfg.max_depth,
        match_margin: cfg.match_margin,
        cleanup_min: cfg.cleanup_min,
        symbol_same: cfg.symbol_same,
    }
}

pub fn graph_spec(cfg: &ReasonerConfig) -> Spec {
    Spec { id: GRAPH_ID.into(), version: 1, seed_entry: "relation-graph-circuit".into(), description: String::new(), circuit: Circuit::RelationGraph(params(cfg)) }
}

pub fn reach_spec(cfg: &ReasonerConfig) -> Spec {
    Spec { id: REACH_ID.into(), version: 1, seed_entry: "relation-reach-circuit".into(), description: String::new(), circuit: Circuit::RelationReach(params(cfg)) }
}

pub const ATOM_ID: &str = "atom-encode";

/// Atom encoder width: names up to `ATOM_BYTES - 1` bytes (longer names are cut).
pub const ATOM_BYTES: usize = 128;
/// Rows of `core.reasoner.atom_table`.
pub const ATOM_BUCKETS: usize = 1024;

pub fn atom_spec(cfg: &ReasonerConfig) -> Spec {
    Spec {
        id: ATOM_ID.into(),
        version: 1,
        seed_entry: "atom-encode-circuit".into(),
        description: String::new(),
        circuit: Circuit::AtomEncode(AtomParams { name_bytes: ATOM_BYTES, buckets: ATOM_BUCKETS, dim: cfg.dim }),
    }
}

/// The seeded random hash table of the atom encoder (`core.reasoner.atom_table`).
pub fn atom_table(cfg: &ReasonerConfig) -> zerolm_tensor::Matrix {
    let mut rng = zerolm_tensor::Rng::from_tag(cfg.seed, "binding.atom_table");
    let mut m = zerolm_tensor::Matrix::zeros(ATOM_BUCKETS, 2 * cfg.dim);
    m.data.iter_mut().for_each(|x| *x = rng.uniform(1.0));
    m
}

/// The compiled atom encoder with its table and a cache of its outputs.
pub struct Atoms {
    pub circuit: Compiled,
    pub table: zerolm_tensor::Matrix,
    cache: std::cell::RefCell<std::collections::HashMap<String, Hv>>,
    recorder: std::cell::RefCell<Option<Vec<Recorded>>>,
    dim: usize,
}

impl Atoms {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        Atoms {
            circuit: compile(&atom_spec(cfg)).expect("valid reasoning config compiles"),
            table: atom_table(cfg),
            cache: Default::default(),
            recorder: Default::default(),
            dim: cfg.dim,
        }
    }

    /// The unit phasor of `namespace:name`, computed by the circuit
    /// (memoized: the cache holds circuit outputs, keyed by the same bytes).
    pub fn encode(&self, namespace: &str, name: &str) -> Hv {
        let key = format!("{namespace}:{name}");
        if let Some(h) = self.cache.borrow().get(&key) {
            return h.clone();
        }
        let mut row = vec![0.0f32; ATOM_BYTES];
        for (slot, b) in row[1..].iter_mut().zip(key.bytes()) {
            *slot = b as f32;
        }
        let inputs = vec![("name", Buf::row(row))];
        let mut w: Vec<(&str, View<'_>)> = vec![("table", view(&self.table, self.table.rows))];
        w.extend(self.circuit.tensors.iter().map(|t| (t.name.as_str(), t.view())));
        let mut b = Bindings::new();
        for (n, t) in &inputs {
            b = b.input(n, t.clone());
        }
        for (n, v) in &w {
            b = b.weight(n, *v);
        }
        let out = isa::run(&self.circuit.program, b).expect("atom circuit runs");
        if let Some(rec) = self.recorder.borrow_mut().as_mut() {
            rec.push(Recorded {
                program: self.circuit.id.clone(),
                bundle: bundle::encode(&self.circuit.program, &w),
                inputs: inputs.into_iter().map(|(n, t)| (n.to_string(), t)).collect(),
                outputs: out.clone(),
            });
        }
        let h = Hv::from_data(self.dim, &out.get("atom").data);
        self.cache.borrow_mut().insert(key, h.clone());
        h
    }
}

pub struct Circuits {
    pub graph: Compiled,
    pub reach: Compiled,
}

impl Circuits {
    pub fn new(cfg: &ReasonerConfig) -> Self {
        Circuits {
            graph: compile(&graph_spec(cfg)).expect("valid reasoning config compiles"),
            reach: compile(&reach_spec(cfg)).expect("valid reasoning config compiles"),
        }
    }
}

/// C5 output for one memory state.
#[derive(Debug, Clone)]
pub struct Graph {
    pub key: (usize, usize, u64, bool, bool),
    pub subj_sim: Buf,
    pub obj_t: Buf,
    pub pos: Buf,
    pub neg: Buf,
}

/// Reached symbols (codebook index) -> premise slots of the shortest proof.
pub type Reach = BTreeMap<usize, Vec<usize>>;

fn view(m: &zerolm_tensor::Matrix, rows: usize) -> View<'_> {
    View::new(rows, m.cols, &m.data[..rows * m.cols])
}

/// One recorded circuit run (evaluation only): the exported bundle (program
/// and every weight exactly as bound), the inputs, and the engine's outputs.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub program: String,
    pub bundle: Vec<u8>,
    pub inputs: Vec<(String, Buf)>,
    pub outputs: Outputs,
}

impl Reasoner {
    fn roles_tensor(&self) -> Vec<f32> {
        (0..ROLES.len()).flat_map(|r| self.role(r).data).collect()
    }

    /// Start (or stop) recording every circuit run for reference replay.
    pub fn record_circuits(&self, on: bool) {
        *self.recorder.borrow_mut() = if on { Some(Vec::new()) } else { None };
        *self.atoms.recorder.borrow_mut() = if on { Some(Vec::new()) } else { None };
    }

    /// Take the runs recorded so far.
    pub fn take_recorded(&self) -> Vec<Recorded> {
        let mut runs = self.recorder.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default();
        runs.extend(self.atoms.recorder.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default());
        runs
    }

    /// Bind, run, and (when recording) export one program run.
    fn exec(&self, c: &Compiled, inputs: Vec<(&str, Buf)>, weights: &[(&str, View<'_>)]) -> Outputs {
        let mut b = Bindings::new();
        for (n, t) in &inputs {
            b = b.input(n, t.clone());
        }
        for (n, v) in weights {
            b = b.weight(n, *v);
        }
        let out = isa::run(&c.program, b).unwrap_or_else(|e| panic!("{} runs on its own memory: {e}", c.id));
        if let Some(rec) = self.recorder.borrow_mut().as_mut() {
            rec.push(Recorded {
                program: c.id.clone(),
                bundle: bundle::encode(&c.program, weights),
                inputs: inputs.into_iter().map(|(n, t)| (n.to_string(), t)).collect(),
                outputs: out.clone(),
            });
        }
        out
    }

    /// The decoded relation graph for the current memory state (cached).
    pub(crate) fn graph(&self) -> std::cell::Ref<'_, Graph> {
        let key = (self.n_slots, self.n_sym, self.clock, self.binding_ablated, self.retrieval_ablated);
        let stale = self.graph_cache.borrow().as_ref().map(|g| g.key != key).unwrap_or(true);
        if stale {
            let roles = self.roles_tensor();
            let zero_meta = vec![0.0f32; self.n_slots * field::COUNT];
            let meta = if self.retrieval_ablated { View::new(self.n_slots, field::COUNT, &zero_meta) } else { view(&self.meta, self.n_slots) };
            let mut w: Vec<(&str, View<'_>)> = vec![
                ("slots", view(&self.slots, self.n_slots)),
                ("meta", meta),
                ("codebook", view(&self.codebook, self.n_sym)),
                ("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles)),
            ];
            w.extend(self.circuits.graph.tensors.iter().map(|t| (t.name.as_str(), t.view())));
            let o = self.exec(&self.circuits.graph, Vec::new(), &w);
            let take = |n: &str| o.values.get(n).cloned().expect("graph output");
            *self.graph_cache.borrow_mut() = Some(Graph { key, subj_sim: take("subj_sim"), obj_t: take("obj_t"), pos: take("pos"), neg: take("neg") });
        }
        std::cell::Ref::map(self.graph_cache.borrow(), |g| g.as_ref().expect("graph computed"))
    }

    /// Operator flags in `OPERATORS` order (evaluation ablations).
    fn op_flags(&self) -> [bool; 8] {
        let o = self.ops;
        [o.class_inclusion, o.transitivity, o.attribute_inheritance, o.comparison, o.arithmetic, o.temporal_ordering, o.negation, o.conditionals]
    }

    /// Run C6 from symbol `from` over positive `rel` edges for operator `op`
    /// (a disabled operator's enable block is bound as zero: direct edges only).
    pub(crate) fn run_reach(&self, from: usize, to: Option<usize>, rel: &Hv, op: &str) -> Option<Outputs> {
        if self.n_slots == 0 || self.n_sym == 0 {
            return None;
        }
        let g = self.graph();
        let roles = self.roles_tensor();
        let opi = OPERATORS.iter().position(|o| *o == op).expect("known operator");
        let mut sel = vec![0.0f32; OPERATORS.len()];
        sel[opi] = 1.0;
        let x = self.code(from);
        let y = to.map(|t| self.code(t)).unwrap_or_else(|| x.clone());
        let inputs = vec![("x", Buf::row(x.data)), ("y", Buf::row(y.data)), ("rel", Buf::row(rel.data.clone())), ("operator", Buf::row(sel))];
        let mut w: Vec<(&str, View<'_>)> = vec![
            ("slots", view(&self.slots, self.n_slots)),
            ("codebook", view(&self.codebook, self.n_sym)),
            ("roles", View::new(ROLES.len(), 2 * self.cfg.dim, &roles)),
            ("subj_sim", g.subj_sim.view()),
            ("obj_t", g.obj_t.view()),
            ("pol", g.pos.view()),
        ];
        let zero = [0.0f32];
        let disabled: Vec<String> =
            OPERATORS.iter().zip(self.op_flags()).filter(|(_, on)| !on).map(|(op, _)| format!("enable_{op}")).collect();
        for t in &self.circuits.reach.tensors {
            let off = disabled.iter().any(|d| *d == t.name);
            w.push((t.name.as_str(), if off { View::new(1, 1, &zero) } else { t.view() }));
        }
        Some(self.exec(&self.circuits.reach, inputs, &w))
    }

    /// Every symbol reachable from `from` with its proof premises.
    pub(crate) fn reach(&self, from: usize, rel: &Hv, op: &str) -> Reach {
        let Some(o) = self.run_reach(from, None, rel, op) else { return Reach::new() };
        let reached = &o.get("reached").data;
        let proofs = o.get("proofs");
        let mut out = Reach::new();
        for (j, &r) in reached.iter().enumerate() {
            if r == 1.0 {
                let row = &proofs.data[j * proofs.cols..(j + 1) * proofs.cols];
                out.insert(j, row.iter().enumerate().filter(|(_, &v)| v > 0.0).map(|(s, _)| s).collect());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerolm_engine::binding::{cosine, sim};

    #[test]
    fn atoms_are_unit_phasors_distinct_and_deterministic() {
        let cfg = ReasonerConfig::default();
        let a = Atoms::new(&cfg);
        let names = ["cat", "cats", "tom", "tomas", "a", "b", "mammal", "animal", "x1", "x2", "x10"];
        let vs: Vec<Hv> = names.iter().map(|n| a.encode("sym", n)).collect();
        for v in &vs {
            for k in 0..cfg.dim {
                let m = v.data[k] * v.data[k] + v.data[cfg.dim + k] * v.data[cfg.dim + k];
                assert!((m - 1.0).abs() < 1e-5, "component {k} has |z|^2 = {m}");
            }
        }
        for i in 0..vs.len() {
            for j in 0..i {
                assert!(cosine(&vs[i], &vs[j]).abs() < 0.15, "{} vs {}: {}", names[i], names[j], cosine(&vs[i], &vs[j]));
            }
        }
        assert!((sim(&vs[0], &vs[0]) - 1.0).abs() < 1e-5);
        let again = Atoms::new(&cfg).encode("sym", "cat");
        assert_eq!(again.data, vs[0].data);
        assert!(cosine(&a.encode("role", "cat"), &vs[0]).abs() < 0.15, "namespaces differ");
    }
}
