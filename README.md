# Don’t Drop It, Compress It: Selective KV Quantization

This repository contains the implementation, experiments, and benchmarking code for our final project in the NLP course *Self-Supervised Methods*. Building on top of the [GPT-Fast](https://github.com/karpathy/gpt-fast) framework, our work investigates efficient memory and inference strategies for large language models (LLMs) through **selective KV cache quantization**.

---

## 🧠 Project Overview

As context lengths in LLMs grow into tens of thousands of tokens, the **Key-Value (KV) cache** becomes a major memory and latency bottleneck—sometimes rivaling the model's own parameter size. Traditional strategies discard old tokens or apply uniform compression, often ignoring token relevance.

We propose a **selective quantization strategy** that:
- **Preserves important tokens** (e.g., recent, high-attention) in high precision (fp16)
- **Aggressively compresses less relevant tokens** (e.g., old, low-attention) using int8 or lower precision

By adapting compression based on token content and age, our goal is to improve:
- **Memory efficiency**
- **Inference latency**
- With **minimal loss in output quality**

---

## 📊 Core Contributions

- ⚙️ **Benchmarking Framework** for memory, latency (TTFT, TPS), and quality (PPL, BLEU, EM/F1)
- 🔁 **Inference Engine Extensions**:
  - Full-precision baseline
  - StreamingLLM-style token eviction
  - Selective quantization for KV cache
- 🔬 **Experimental Evaluation** on long-context benchmarks like NarrativeQA and PG-19
- 🔄 Compatibility with multiple LLM backends (TinyLlama, LLaMA-2, Mistral)
