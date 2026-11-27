//! Sentence embeddings with candle: ModernBERT (granite-embedding-small-english-r2), CLS pooling,
//! L2-normalised. Runs on Metal when available.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::Config;

use crate::modernbert::ModernBert;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::config::{self, EMBED_TOKENS, MODEL, MODEL_FILES};

pub struct Embedder {
    model: ModernBert,
    tok: Tokenizer,
    device: Device,
    pub dim: usize,
}

fn hf_cache_dir(model: &str) -> Option<PathBuf> {
    let base = std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")))?;
    let snaps = base.join(format!("models--{}", model.replace('/', "--"))).join("snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path())
        .find(|p| MODEL_FILES.iter().all(|f| p.join(f).exists()))
}

/// Model directory: our own copy, else the Hugging Face cache, else download.
pub fn model_dir() -> Result<PathBuf> {
    model_dir_for(MODEL)
}

pub fn model_dir_for(model: &str) -> Result<PathBuf> {
    let own = config::home().join("model").join(model.replace('/', "--"));
    if MODEL_FILES.iter().all(|f| own.join(f).exists()) {
        return Ok(own);
    }
    if let Some(p) = hf_cache_dir(model) {
        return Ok(p);
    }
    std::fs::create_dir_all(&own)?;
    for f in MODEL_FILES {
        let url = format!("https://huggingface.co/{model}/resolve/main/{f}");
        eprintln!("downloading {url}");
        let mut resp = ureq::get(&url).call().with_context(|| url.clone())?;
        let mut out = std::fs::File::create(own.join(format!("{f}.part")))?;
        std::io::copy(&mut resp.body_mut().as_reader(), &mut out)?;
        std::fs::rename(own.join(format!("{f}.part")), own.join(f))?;
    }
    Ok(own)
}

impl Embedder {
    pub fn load() -> Result<Self> {
        Self::load_from(&model_dir()?)
    }

    pub fn load_from(dir: &Path) -> Result<Self> {
        Self::load_on(dir, device())
    }

    pub fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let mut tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        tok.with_truncation(Some(TruncationParams { max_length: EMBED_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!("{e}"))?;
        tok.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: cfg.pad_token_id,
            ..Default::default()
        }));
        let vb = load_weights(dir, &device)?;
        let model = ModernBert::load(&vb, &cfg)?;
        Ok(Embedder { model, tok, device, dim: cfg.hidden_size })
    }

    /// Embed many texts in batches of about `budget` padded tokens. All texts are tokenized
    /// once, ordered by their real token count and cut into batches that pad only to their own
    /// longest member: one long-line chunk no longer pads a whole batch to the maximum.
    /// Vectors come back in input order.
    pub fn encode_budget(&self, texts: &[String], budget: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let enc = self.tok.encode_batch(texts.to_vec(), true).map_err(|e| anyhow!("{e}"))?;
        let len = |e: &tokenizers::Encoding| e.get_attention_mask().iter().filter(|&&m| m != 0).count();
        let mut order: Vec<usize> = (0..enc.len()).collect();
        order.sort_by_key(|&i| len(&enc[i]));
        let mut out = vec![vec![]; enc.len()];
        let mut start = 0;
        while start < order.len() {
            // Sorted ascending: the last row of a batch is its longest.
            let mut end = start + 1;
            while end < order.len() && len(&enc[order[end]]) * (end + 1 - start) <= budget {
                end += 1;
            }
            let part = &order[start..end];
            let n = len(&enc[*part.last().unwrap()]).max(1);
            let ids: Vec<u32> = part.iter().flat_map(|&i| enc[i].get_ids()[..n].iter().copied()).collect();
            let lens: Vec<usize> = part.iter().map(|&i| len(&enc[i])).collect();
            let ids = Tensor::from_vec(ids, (part.len(), n), &self.device)?;
            let cls = self.model.forward_cls(&ids, &lens)?;
            let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
            for (&i, v) in part.iter().zip(cls.broadcast_div(&norm)?.to_vec2::<f32>()?) {
                out[i] = v;
            }
            start = end;
        }
        Ok(out)
    }

    pub fn encode(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let enc = self.tok.encode_batch(texts.to_vec(), true).map_err(|e| anyhow!("{e}"))?;
        let (ids, lens) = batch_ids(&enc, &self.device)?;
        let cls = self.model.forward_cls(&ids, &lens)?; // (b, d) F32
        let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
        let cls = cls.broadcast_div(&norm)?;
        Ok(cls.to_vec2::<f32>()?)
    }
}

/// Metal when present; `REFX_DEVICE=cpu` forces the CPU.
pub fn device() -> Device {
    if std::env::var("REFX_DEVICE").is_ok_and(|d| d == "cpu") {
        return Device::Cpu;
    }
    Device::new_metal(0).unwrap_or(Device::Cpu)
}

/// Token ids `(batch, seq)` of a right-padded batch and each row's unpadded length.
pub fn batch_ids(enc: &[tokenizers::Encoding], device: &Device) -> Result<(Tensor, Vec<usize>)> {
    let (b, n) = (enc.len(), enc[0].get_ids().len());
    let ids: Vec<u32> = enc.iter().flat_map(|e| e.get_ids().iter().copied()).collect();
    let lens = enc.iter().map(|e| e.get_attention_mask().iter().filter(|&&m| m != 0).count()).collect();
    Ok((Tensor::from_vec(ids, (b, n), device)?, lens))
}

/// F16 on Metal, F32 on the CPU (whose F16 matmul is slow); `REFX_DTYPE=f32` forces F32.
pub fn dtype_for(device: &Device) -> DType {
    if device.is_metal() && std::env::var("REFX_DTYPE").map_or(true, |d| d != "f32") {
        DType::F16
    } else {
        DType::F32
    }
}

/// Load safetensors by reading them into memory (no mmap, which candle exposes as `unsafe`).
pub fn load_weights(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let bytes = std::fs::read(dir.join("model.safetensors"))?;
    let tensors = candle_core::safetensors::load_buffer(&bytes, device)?;
    let dtype = dtype_for(device);
    let tensors = tensors
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_dtype(dtype)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    Ok(VarBuilder::from_tensors(tensors, dtype, device))
}
