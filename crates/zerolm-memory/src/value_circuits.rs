//! The compiled value codec and `W_FAST` circuits bound to a store
//! (`zerolm_wsp::circuits::value_codec`): `value-encode` turns value bytes
//! into code rows, `value-match` decodes a stored row and compares it with a
//! candidate value, `fast-recall` and `fast-write` read and update the
//! short-lived associative matrix `memory.fast`. The host copies bytes into
//! input rows, binds model tensors, and reads outputs; it inspects no byte.

use crate::persistent_memory::MemoryConfig;
use zerolm_engine::isa::{self, bundle, Bindings, Buf, Outputs, View};
use zerolm_wsp::circuits::value_codec::{CodecParams, FastParams};
use zerolm_wsp::{compile, Circuit, Compiled, Spec};

pub const ENCODE_ID: &str = "value-encode";
pub const MATCH_ID: &str = "value-match";
pub const RECALL_ID: &str = "fast-recall";
pub const WRITE_ID: &str = "fast-write";
/// Normalization epsilon of the fast-weight delta rule.
pub const FAST_EPS: f32 = 1e-6;

fn codec(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> CodecParams {
    CodecParams { positions: cfg.positions(), code_rows, code_dim }
}

fn fast(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> FastParams {
    FastParams {
        codec: codec(cfg, code_rows, code_dim),
        key_dim: cfg.key_dim,
        expansion: cfg.fast_expansion,
        active: cfg.fast_active,
        lr: cfg.fast_lr,
        decay: cfg.fast_decay,
        eps: FAST_EPS,
    }
}

fn spec(id: &str, circuit: Circuit) -> Spec {
    Spec { id: id.into(), version: 1, seed_entry: "value-codec-circuits".into(), description: String::new(), circuit }
}

/// The four specs for a configuration and code table shape (equal to the
/// shipped specs for the frozen default configuration).
pub fn specs(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> [Spec; 4] {
    [
        spec(ENCODE_ID, Circuit::ValueEncode(codec(cfg, code_rows, code_dim))),
        spec(MATCH_ID, Circuit::ValueMatch(codec(cfg, code_rows, code_dim))),
        spec(RECALL_ID, Circuit::FastRecall(fast(cfg, code_rows, code_dim))),
        spec(WRITE_ID, Circuit::FastWrite(fast(cfg, code_rows, code_dim))),
    ]
}

/// A value decoded by a circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub bytes: Vec<u8>,
    /// Smallest top-1 minus top-2 code score over the decoded positions (up
    /// to the first NUL): how far the read is from flipping to another byte.
    pub margin: f32,
}

/// One recorded circuit run (evaluation only): the exported bundle (program
/// and every weight as bound), the inputs (including initial state), and the
/// engine's outputs and final state.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub program: String,
    pub bundle: Vec<u8>,
    pub inputs: Vec<(String, Buf)>,
    pub outputs: Outputs,
}

pub struct ValueCircuits {
    pub encode: Compiled,
    pub matcher: Compiled,
    pub recall: Compiled,
    pub write: Compiled,
    positions: usize,
    recorder: std::cell::RefCell<Option<Vec<Recorded>>>,
}

impl ValueCircuits {
    pub fn new(cfg: &MemoryConfig, code_rows: usize, code_dim: usize) -> Self {
        let [e, m, r, w] = specs(cfg, code_rows, code_dim).map(|s| compile(&s).expect("valid memory configuration compiles"));
        ValueCircuits { encode: e, matcher: m, recall: r, write: w, positions: cfg.positions(), recorder: Default::default() }
    }

    pub fn all(&self) -> [&Compiled; 4] {
        [&self.encode, &self.matcher, &self.recall, &self.write]
    }

    /// Value bytes copied into a NUL-padded row (data movement only). The
    /// caller has checked the length and that no byte is NUL.
    pub fn value_input(&self, value: &str) -> Buf {
        let mut row = vec![0.0f32; self.positions];
        for (slot, b) in row.iter_mut().zip(value.bytes()) {
            *slot = b as f32;
        }
        Buf::row(row)
    }

    pub fn record(&self, on: bool) {
        *self.recorder.borrow_mut() = if on { Some(Vec::new()) } else { None };
    }

    pub fn take_recorded(&self) -> Vec<Recorded> {
        self.recorder.borrow_mut().as_mut().map(std::mem::take).unwrap_or_default()
    }

    fn run(&self, c: &Compiled, inputs: Vec<(&str, Buf)>, weights: &[(&str, View<'_>)], state: Option<(&str, Buf)>) -> Outputs {
        let mut w: Vec<(&str, View<'_>)> = weights.to_vec();
        w.extend(c.tensors.iter().map(|t| (t.name.as_str(), t.view())));
        let mut b = Bindings::new();
        for (n, t) in &inputs {
            b = b.input(n, t.clone());
        }
        if let Some((n, t)) = &state {
            b = b.state(n, t.clone());
        }
        for (n, v) in &w {
            b = b.weight(n, *v);
        }
        let out = isa::run(&c.program, b).unwrap_or_else(|e| panic!("{} runs on its own store: {e}", c.id));
        if let Some(rec) = self.recorder.borrow_mut().as_mut() {
            let mut all: Vec<(String, Buf)> = inputs.into_iter().map(|(n, t)| (n.to_string(), t)).collect();
            all.extend(state.map(|(n, t)| (n.to_string(), t)));
            rec.push(Recorded { program: c.id.clone(), bundle: bundle::encode(&c.program, &w), inputs: all, outputs: out.clone() });
        }
        out
    }

    /// Code row of a value (`value-encode`).
    pub fn encode_value(&self, codes: View<'_>, value: &str) -> Vec<f32> {
        let o = self.run(&self.encode, vec![("value", self.value_input(value))], &[("codes", codes)], None);
        o.get("row").data.clone()
    }

    /// Decode a stored row and compare it with `value` (`value-match`).
    pub fn matches(&self, codes: View<'_>, row: &[f32], value: &str) -> (Decoded, bool) {
        let o = self.run(&self.matcher, vec![("row", Buf::row(row.to_vec())), ("value", self.value_input(value))], &[("codes", codes)], None);
        (decoded(&o), o.scalar("same") == 1.0)
    }

    /// Recall from `W_FAST` (`fast-recall`).
    pub fn recall_fast(&self, codes: View<'_>, separator: View<'_>, fast: View<'_>, key: &[f32]) -> Decoded {
        let o = self.run(&self.recall, vec![("key", Buf::row(key.to_vec()))], &[("separator", separator), ("fast", fast), ("codes", codes)], None);
        decoded(&o)
    }

    /// Write one association into `W_FAST` (`fast-write`); returns the new matrix.
    pub fn write_fast(&self, codes: View<'_>, separator: View<'_>, fast: Buf, key: &[f32], value: &str) -> Buf {
        let mut o = self.run(
            &self.write,
            vec![("key", Buf::row(key.to_vec())), ("value", self.value_input(value))],
            &[("separator", separator), ("codes", codes)],
            Some(("fast", fast)),
        );
        o.state.remove("fast").expect("fast-write returns its state")
    }
}

fn decoded(o: &Outputs) -> Decoded {
    Decoded { bytes: crate::read_circuit::bytes(o.get("bytes")), margin: o.scalar("margin") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerolm_tensor::Matrix;

    fn v(m: &Matrix) -> View<'_> {
        View::new(m.rows, m.cols, &m.data)
    }

    fn setup() -> (MemoryConfig, Matrix, Matrix, ValueCircuits) {
        let cfg = MemoryConfig::default();
        let codes = zerolm_engine::bootstrap::default_model(cfg.seed).embed;
        let sep = Matrix::uniform(cfg.fast_expansion, cfg.key_dim, cfg.seed, crate::persistent_memory::SEPARATOR, 1.0);
        let c = ValueCircuits::new(&cfg, codes.rows, codes.cols);
        (cfg, codes, sep, c)
    }

    #[test]
    fn encoded_rows_are_the_code_rows_and_match_decodes_them_back() {
        let (cfg, codes, _, c) = setup();
        let row = c.encode_value(v(&codes), "blue");
        assert_eq!(row.len(), cfg.positions() * codes.cols);
        for (p, b) in b"blue\0\0".iter().enumerate() {
            assert_eq!(&row[p * codes.cols..(p + 1) * codes.cols], codes.row(*b as usize), "position {p}");
        }
        let (d, same) = c.matches(v(&codes), &row, "blue");
        assert!(same && d.bytes == b"blue" && d.margin > 0.0);
        for other in ["blu", "blues", "bluf", "x"] {
            assert!(!c.matches(v(&codes), &row, other).1, "{other}");
        }
        // A superposition decodes to its dominant value.
        let green = c.encode_value(v(&codes), "green");
        let mix: Vec<f32> = row.iter().zip(&green).map(|(a, b)| 0.9 * a + 0.1 * b).collect();
        assert!(c.matches(v(&codes), &mix, "blue").1);
    }

    #[test]
    fn fast_writes_are_recalled_and_fade_with_decay() {
        let (cfg, codes, sep, c) = setup();
        let key = |i: u64| {
            let mut r = zerolm_tensor::Rng::from_tag(i, "test-key");
            (0..cfg.key_dim).map(|_| r.uniform(1.0)).collect::<Vec<f32>>()
        };
        let mut m = Buf::zeros(cfg.positions() * codes.cols, cfg.fast_expansion);
        m = c.write_fast(v(&codes), v(&sep), m, &key(1), "amber");
        let read = |m: &Buf, i| c.recall_fast(v(&codes), v(&sep), View::new(m.rows, m.cols, &m.data), &key(i)).bytes;
        assert_eq!(read(&m, 1), b"amber");
        let once = m.clone();
        for i in 2..400 {
            m = c.write_fast(v(&codes), v(&sep), m, &key(i), &format!("v{i}"));
        }
        assert_ne!(read(&m, 1), b"amber", "decayed after 400 later writes");
        assert_eq!(read(&m, 399), b"v399", "the newest association is recalled");
        assert_ne!(once.data, m.data);
    }
}
