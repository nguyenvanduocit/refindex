//! Cross-encoder reranking: granite-embedding-reranker-english-r2 reads (query, chunk) together
//! and scores relevance. The retrieval lists only compare a query vector with chunk vectors and
//! count words; the cross-encoder fixes their "same topic, wrong answer" top 10.
//!
//! candle's `ModernBertForSequenceClassification` is not used: its head hard-codes GELU (this
//! model's `classifier_activation` is SiLU), it softmaxes a single label to 1.0, and its config
//! cannot read integer `label2id`. The backbone is candle's; the head follows Hugging Face's
//! `ModernBertForSequenceClassification`: CLS -> dense (no bias) -> activation -> LayerNorm
//! (no bias) -> linear -> sigmoid.

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use candle_transformers::models::modernbert::Config;

use crate::modernbert::ModernBert;
use tokenizers::{EncodeInput, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::embed;

pub const MODEL: &str = "ibm-granite/granite-embedding-reranker-english-r2";
pub const MAX_TOKENS: usize = 256;
const BATCH: usize = 16;

pub struct Reranker {
    model: ModernBert,
    dense: Linear,
    norm: LayerNorm,
    classifier: Linear,
    silu: bool,
    tok: Tokenizer,
    device: Device,
}

impl Reranker {
    pub fn load() -> Result<Self> {
        let dir = embed::model_dir_for(MODEL)?;
        let device = embed::device();
        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let act = raw["classifier_activation"].as_str().unwrap_or("gelu").to_string();
        let obj = raw.as_object_mut().ok_or_else(|| anyhow!("bad config"))?;
        for k in ["id2label", "label2id", "classifier_pooling"] {
            obj.remove(k); // candle's classifier config cannot read these; the head is ours
        }
        let cfg: Config = serde_json::from_value(raw)?;
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        tok.with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        tok.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: cfg.pad_token_id,
            ..Default::default()
        }));
        let vb: VarBuilder = embed::load_weights(&dir, &device)?;
        let model = ModernBert::load(&vb.pp("model"), &cfg)?;
        // The head is tiny and reads the F32 CLS state: keep it in F32.
        let h = cfg.hidden_size;
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        let dense = Linear::new(f32(vb.get((h, h), "head.dense.weight")?)?, None);
        let norm = LayerNorm::new_no_bias(f32(vb.get(h, "head.norm.weight")?)?, cfg.layer_norm_eps);
        let classifier = Linear::new(
            f32(vb.get((1, h), "classifier.weight")?)?,
            vb.get(1, "classifier.bias").ok().map(f32).transpose()?,
        );
        Ok(Reranker { model, dense, norm, classifier, silu: act == "silu", tok, device })
    }

    /// Relevance in [0, 1] of each document to the query.
    pub fn score(&self, query: &str, docs: &[String]) -> Result<Vec<f32>> {
        // Tokenize once, then batch by length so a batch pads to its own longest pair, not to
        // the longest of all.
        let pairs: Vec<EncodeInput> =
            docs.iter().map(|d| EncodeInput::Dual(query.into(), d.as_str().into())).collect();
        let all = self.tok.encode_batch(pairs, false).map_err(|e| anyhow!("{e}"))?;
        let mut order: Vec<usize> = (0..docs.len()).collect();
        // The tokenizer pads every batch, this one included: length = unmasked tokens.
        order.sort_by_key(|&i| all[i].get_attention_mask().iter().filter(|&&m| m != 0).count());
        let mut out = vec![0f32; docs.len()];
        for part in order.chunks(BATCH) {
            let pairs: Vec<EncodeInput> =
                part.iter().map(|&i| EncodeInput::Dual(query.into(), docs[i].as_str().into())).collect();
            let t = std::time::Instant::now();
            let enc = self.tok.encode_batch(pairs, true).map_err(|e| anyhow!("{e}"))?;
            let (b, n) = (enc.len(), enc[0].get_ids().len());
            let (ids, lens) = embed::batch_ids(&enc, &self.device)?;
            let cls = self.model.forward_cls(&ids, &lens)?; // (b, d) F32
            let x = self.dense.forward(&cls)?;
            let x = if self.silu { candle_nn::ops::silu(&x)? } else { x.gelu_erf()? };
            let logits = self.classifier.forward(&self.norm.forward(&x)?)?; // (b, 1)
            let p = candle_nn::ops::sigmoid(&logits)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            if std::env::var_os("REFX_TIMING").is_some() {
                eprintln!("rerank batch {b}x{n}: {:?}", t.elapsed());
            }
            for (&i, s) in part.iter().zip(p) {
                out[i] = s;
            }
        }
        Ok(out)
    }
}
