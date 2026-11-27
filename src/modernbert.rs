//! ModernBERT encoder for refx: only what a CLS-pooled embedder and a cross-encoder need, built
//! for throughput on Metal. Same weights and the same function as Hugging Face's `ModernBertModel`
//! (checked by `referee/embed_referee.py` and `referee/rerank_referee.py`); differs from
//! candle's `models::modernbert` in how it computes:
//!
//! - **Half precision on Metal.** Weights and activations are F16 (norms and the attention
//!   softmax accumulate in F32 inside their kernels); the CPU path stays F32.
//! - **Fused attention.** One `sdpa` kernel per layer (scale, mask, softmax and the value
//!   product), instead of a matmul, a broadcast add, a three-kernel softmax and another matmul.
//! - **Masks built once per batch,** shared by all heads through stride-0 broadcasting. Masked
//!   scores are a finite -30000, not -inf: a padded query whose whole sliding window is padding
//!   would otherwise softmax to NaN, and NaN values leak into real rows through `0 * NaN`.
//! - **Only what CLS depends on.** The output is the first token. Walking back from the last
//!   layer, a sliding-window layer needs its queries only for the tokens the next layer reads
//!   and its keys only one half-window further; a global layer needs everything. So trailing
//!   local layers run on a prefix of the sequence (in granite-small the last two layers are
//!   local: the last one computes a single query over 65 keys).
//! - **Rotary embedding in place on the `(batch, seq, head, dim)` layout** (`rope_thd`), with
//!   cos/sin tables built once at load.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};
use candle_transformers::models::modernbert::Config;

/// Masked attention score: finite so a fully masked row stays finite (see module docs);
/// two of them still fit in F16.
const MASKED: f64 = -30000.0;

struct Layer {
    attn_norm: Option<LayerNorm>,
    wqkv: Tensor,
    wo: Tensor,
    mlp_norm: LayerNorm,
    wi: Tensor,
    wo_mlp: Tensor,
    local: bool,
}

pub struct ModernBert {
    emb: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    /// (cos, sin) for global and local layers: (max_positions, head_dim / 2).
    rope_global: (Tensor, Tensor),
    rope_local: (Tensor, Tensor),
    heads: usize,
    head_dim: usize,
    half_window: usize,
    pub hidden: usize,
    pub dtype: DType,
    device: Device,
}

fn norm(vb: &VarBuilder, name: &str, cfg: &Config) -> Result<LayerNorm> {
    Ok(LayerNorm::new_no_bias(vb.get(cfg.hidden_size, &format!("{name}.weight"))?, cfg.layer_norm_eps))
}

fn rope_tables(cfg: &Config, theta: f64, dim: usize, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
    let inv: Vec<f32> = (0..dim).step_by(2).map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32).collect();
    let n = cfg.max_position_embeddings;
    let mut freqs = Vec::with_capacity(n * inv.len());
    for t in 0..n {
        freqs.extend(inv.iter().map(|f| t as f32 * f));
    }
    let f = Tensor::from_vec(freqs, (n, inv.len()), dev)?;
    Ok((f.cos()?.to_dtype(dtype)?, f.sin()?.to_dtype(dtype)?))
}

/// `x @ w.T` for `x: (n, in)` and a row-major `w: (out, in)`.
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?)
}

impl ModernBert {
    /// `vb` is rooted at the backbone (`embeddings.*`, `layers.*`, `final_norm.*`) and already
    /// holds `dtype`.
    pub fn load(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let (d, h) = (cfg.hidden_size, cfg.num_attention_heads);
        let head_dim = d / h;
        let dev = vb.device().clone();
        let dtype = vb.dtype();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            layers.push(Layer {
                attn_norm: norm(vb, &format!("{p}.attn_norm"), cfg).ok(), // identity in layer 0
                wqkv: vb.get((3 * d, d), &format!("{p}.attn.Wqkv.weight"))?,
                wo: vb.get((d, d), &format!("{p}.attn.Wo.weight"))?,
                mlp_norm: norm(vb, &format!("{p}.mlp_norm"), cfg)?,
                wi: vb.get((2 * cfg.intermediate_size, d), &format!("{p}.mlp.Wi.weight"))?,
                wo_mlp: vb.get((d, cfg.intermediate_size), &format!("{p}.mlp.Wo.weight"))?,
                local: i % cfg.global_attn_every_n_layers != 0,
            });
        }
        Ok(ModernBert {
            emb: Embedding::new(vb.get((cfg.vocab_size, d), "embeddings.tok_embeddings.weight")?, d),
            emb_norm: norm(vb, "embeddings.norm", cfg)?,
            layers,
            final_norm: norm(vb, "final_norm", cfg)?,
            rope_global: rope_tables(cfg, cfg.global_rope_theta, head_dim, dtype, &dev)?,
            rope_local: rope_tables(cfg, cfg.local_rope_theta, head_dim, dtype, &dev)?,
            heads: h,
            head_dim,
            half_window: cfg.local_attention / 2,
            hidden: d,
            dtype,
            device: dev,
        })
    }

    /// Hidden state of the first (CLS) token after the final norm, `(batch, hidden)` in F32.
    /// `ids` is `(batch, seq)`, right-padded; `lens[b]` is the unpadded length of row `b`.
    pub fn forward_cls(&self, ids: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        // How many leading tokens each layer must output, walking back from CLS.
        let mut need_out = vec![0; self.layers.len()];
        let mut need = 1;
        for (i, l) in self.layers.iter().enumerate().rev() {
            need_out[i] = need;
            need = if l.local { (need + self.half_window).min(s) } else { s };
        }
        // Key padding (batch, 1, 1, seq) and the sliding window (1, 1, seq, seq), additive.
        let pad: Vec<f32> = lens.iter()
            .flat_map(|&n| (0..s).map(move |j| if j < n { 0.0 } else { MASKED as f32 }))
            .collect();
        let pad = Tensor::from_vec(pad, (b, 1, 1, s), &self.device)?.to_dtype(self.dtype)?;
        let hw = self.half_window;
        let win: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if i.abs_diff(j) > hw { MASKED as f32 } else { 0.0 }))
            .collect();
        let win = Tensor::from_vec(win, (1, 1, s, s), &self.device)?.to_dtype(self.dtype)?;
        let local_mask = pad.broadcast_add(&win)?; // (b, 1, s, s)

        let mut x = self.emb_norm.forward(&self.emb.forward(ids)?)?; // (b, s, d)
        for (i, l) in self.layers.iter().enumerate() {
            let kv = x.dim(1)?; // tokens this layer reads
            let q_len = need_out[i];
            let h = match &l.attn_norm {
                Some(n) => n.forward(&x)?,
                None => x.clone(),
            };
            let qkv = linear(&h.reshape((b * kv, self.hidden))?, &l.wqkv)?
                .reshape((b, kv, 3, self.heads, self.head_dim))?;
            let (cos, sin) = if l.local { &self.rope_local } else { &self.rope_global };
            let rope = |t: Tensor| -> Result<Tensor> {
                Ok(candle_nn::rotary_emb::rope_thd(&t.contiguous()?, cos, sin)?.transpose(1, 2)?.contiguous()?)
            };
            let q = rope(qkv.i((.., ..q_len, 0))?)?; // (b, h, q_len, hd)
            let k = rope(qkv.i((.., .., 1))?)?; // (b, h, kv, hd)
            let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
            let mask = if l.local {
                local_mask.i((.., .., ..q_len, ..kv))?
            } else {
                pad.i((.., .., .., ..kv))?.broadcast_as((b, 1, q_len, kv))?
            };
            let mask = mask.broadcast_as((b, self.heads, q_len, kv))?;
            let scale = (self.head_dim as f32).powf(-0.5);
            let att = self.attention(&q, &k, &v, &mask, scale)?; // (b, h, q_len, hd)
            let att = att.transpose(1, 2)?.contiguous()?.reshape((b * q_len, self.hidden))?;
            let att = linear(&att, &l.wo)?.reshape((b, q_len, self.hidden))?;
            let x_res = if q_len < kv { x.narrow(1, 0, q_len)? } else { x };
            let x1 = (x_res + att)?;
            let m = l.mlp_norm.forward(&x1)?.reshape((b * q_len, self.hidden))?;
            let gi = linear(&m, &l.wi)?;
            let half = gi.dim(1)? / 2;
            let act = (gi.narrow(1, 0, half)?.gelu_erf()? * gi.narrow(1, half, half)?)?; // GeGLU
            let mlp = linear(&act, &l.wo_mlp)?.reshape((b, q_len, self.hidden))?;
            x = (x1 + mlp)?;
        }
        let cls = x.i((.., 0, ..))?.contiguous()?;
        Ok(self.final_norm.forward(&cls)?.to_dtype(DType::F32)?)
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f32) -> Result<Tensor> {
        // candle's single-query (vector) sdpa kernel ignores the mask, so one query takes the
        // plain path; it is one row per head and costs nothing.
        if self.device.is_metal() && q.dim(2)? > 1 {
            return Ok(candle_nn::ops::sdpa(q, k, v, Some(mask), false, scale, 1.0)?);
        }
        let att = (q.matmul(&k.t()?)? * scale as f64)?.broadcast_add(mask)?;
        Ok(candle_nn::ops::softmax_last_dim(&att)?.matmul(v)?)
    }
}
