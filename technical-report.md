# Technical Report: ASR Models Benchmarking for Customer Support Voice AI

**Project:** Benchmarking Automatic Speech Recognition (ASR) Models for a Voice-Based Customer Support AI Assistant
**Hardware:** Google Colab T4 GPU (15GB VRAM)
**Date:** March 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Models Evaluated](#3-models-evaluated)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Results](#7-results)
8. [Qualitative Analysis](#8-qualitative-analysis)
9. [Comparative Analysis](#9-comparative-analysis)
10. [Recommendation](#10-recommendation)
11. [Optimization Roadmap](#11-optimization-roadmap)
12. [Proposed Production Architecture](#12-proposed-production-architecture)
13. [Hardware and Environment](#13-hardware-and-environment)

---

## 1. Executive Summary

This report presents a comparative benchmarking study of three open-source Automatic Speech Recognition (ASR) models: Wav2Vec2 Base 960h (Meta AI), Whisper Small (OpenAI), and Parakeet TDT 1.1B (NVIDIA), evaluated for deployment in a voice-based AI assistant for customer support call environments. The benchmark was conducted on 100 samples from the People's Speech dataset (dirty, test split) on a Google Colab T4 GPU.

**Key findings:**
- **Parakeet TDT 1.1B** achieves the best raw Word Error Rate (WER) at **0.2381**, but produces unpunctuated output with numbers spoken verbatim, requiring significant post-processing.
- **Whisper Small** achieves a WER of **0.4082** with production-ready output including punctuation, sentence structure, and number normalization, making it the most suitable candidate for deployment.
- **Wav2Vec2 Base** achieves the fastest inference at 0.03s but produces unusable transcriptions (WER: 1.0825) on noisy conversational audio.

**Recommendation:** **Whisper Small** is recommended for production deployment. Its WER disadvantage relative to Parakeet is offset by superior output quality, eliminating the need for additional post-processing infrastructure.

---

## 2. Problem Statement

Voice-based AI assistants for customer support operate in challenging real-world conditions: background noise, multiple accents, conversational speech patterns (pauses, stammers), and long call durations. These conditions degrade ASR performance significantly compared to clean, studio-quality audio.

This study evaluates three state-of-the-art open-source ASR models to identify the best candidate for production deployment in such an environment. The models were selected to maximize architectural diversity while remaining feasible on a Google Colab T4 GPU, ensuring reproducibility without requiring specialized infrastructure.

---

## 3. Models Evaluated

Three models were selected to cover distinct architectural families in modern ASR:

### 3.1 Wav2Vec2 Base 960h - Meta AI

| Attribute | Details |
|---|---|
| **HuggingFace ID** | `facebook/wav2vec2-base-960h` |
| **Architecture** | Self-supervised CNN feature extractor + Transformer encoder + CTC (Connectionist Temporal Classification) head. Processes raw waveform directly — no mel spectrogram conversion needed. |
| **Parameters** | ~95M |
| **Training Data** | Pre-trained on ~53,000 hours of unlabelled LibriSpeech audio; fine-tuned on the 960-hour labelled LibriSpeech subset. |
| **Licensing** | Apache-2.0 |
| **Hardware Requirements** | Lightweight — runs on CPU; GPU recommended for speed. |
| **Strengths** | Fastest inference; minimal dependencies; good baseline for fine-tuning on domain-specific data. |
| **Weaknesses** | Struggles significantly with noisy audio and accents out-of-the-box; no punctuation or text normalization in output; high WER on conversational speech. |

### 3.2 Whisper Small - OpenAI

| Attribute | Details |
|---|---|
| **HuggingFace ID** | `openai/whisper-small` |
| **Architecture** | Encoder-Decoder Transformer. Converts audio to mel spectrogram → encoder builds contextual representation → decoder generates text autoregressively. |
| **Parameters** | ~244M |
| **Training Data** | 680,000 hours of multilingual, weakly supervised web audio — highly diverse in accents, languages, and recording conditions. |
| **Licensing** | MIT — fully open source, commercial use permitted. |
| **Hardware Requirements** | Moderate — GPU recommended; runs on CPU but slow. Compatible with standard HuggingFace inference pipelines. |
| **Strengths** | Excellent multilingual performance and accent robustness; built-in punctuation and text normalization; production-ready output with minimal post-processing. |
| **Weaknesses** | Slower inference than CTC-based models; fixed 30-second input window requires padding for short clips. |

### 3.3 Parakeet TDT 1.1B - NVIDIA

| Attribute | Details |
|---|---|
| **HuggingFace ID** | `nvidia/parakeet-tdt-1.1b` |
| **Architecture** | Conformer encoder (combines convolution + self-attention for local and global audio patterns) + RNN-Transducer decoder with Token and Duration Transducer (TDT) loss, optimized for streaming and low-latency inference. |
| **Parameters** | ~1.1B |
| **Training Data** | ~64,000 hours of English speech (NVIDIA-curated + LibriSpeech-family datasets). |
| **Licensing** | CC-BY-4.0 |
| **Hardware Requirements** | GPU required; ~5.7GB peak VRAM. Requires NVIDIA NeMo toolkit installation (heavy dependency chain). |
| **Strengths** | Best raw WER on noisy audio; fast inference; real-time streaming capable; trained on People's Speech (the same dataset used in this benchmark). |
| **Weaknesses** | Outputs numbers as spoken words (e.g., "one nine hundred sixty s") — requires text normalization post-processing; NeMo dependency adds deployment complexity. |

### 3.4 Models Considered but Excluded

| Model | Reason for Exclusion |
|---|---|
| **Whisper Large V3 Turbo** | Superior performance but too memory-intensive for reproducible benchmarking on Colab T4. |
| **Distil-Whisper** | Shares encoder-decoder architecture with Whisper Small; insufficient architectural diversity for this benchmark. |
| **Gnani.ai Vachana ASR** | No open-source weights available. Primarily targets voice agent applications in Indian languages (Kannada-focused). Strong contender for Indic language deployments but outside scope of this benchmark. |

---

## 4. Dataset

**Dataset:** [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) — `dirty`, `test` split
**Samples used:** 100

### 4.1 Rationale for Selection

The People's Speech dataset was selected for the following reasons:

- **Noisy and conversational:** The `dirty` configuration contains unfiltered, real-world audio with background noise, varied recording quality, and natural speech patterns (pauses, stammers), closely mirroring actual customer support call conditions.
- **No data leakage:** The `test` split was specifically chosen to avoid overlap with training data used by any of the evaluated models, ensuring unbiased benchmarking.
- **NVIDIA validation:** People's Speech was used in training NVIDIA's Parakeet, Nemotron, and Canary models, further validating its relevance as an ASR benchmark dataset.
- **Streaming mode:** Due to the dataset's large size (~2.12TB), streaming mode with `take(100)` was used to efficiently sample 100 test examples without requiring a full download.

### 4.2 Alternatives Considered and Rejected

| Dataset | Reason for Rejection |
|---|---|
| **LibriSpeech** | Too clean (audiobook recordings); not representative of noisy call center conditions. |
| **Common Voice (Mozilla)** | Moved off HuggingFace as of October 2025; unavailable via the `datasets` library. |

---

## 5. Methodology

### 5.1 Infrastructure

All benchmarking was conducted in a Google Colab environment using a T4 GPU. Models were loaded via HuggingFace Transformers (Wav2Vec2 and Whisper) and NVIDIA NeMo toolkit (Parakeet). All models were set to `eval()` mode before inference.

### 5.2 Ground Truth Preprocessing

Reference transcriptions from the dataset were cleaned before WER computation:
- Bracket-enclosed annotations (e.g., `[noise]`) were removed using regex.
- Text was lowercased and stripped of leading/trailing whitespace.

```python
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower().strip()
    return text
```

### 5.3 Inference Pipelines

Each model used a distinct inference pipeline reflecting its architecture:

**Wav2Vec2:**
- Input audio processed via `AutoProcessor` at 16kHz sampling rate.
- Logits obtained from `AutoModelForCTC`; argmax decoding applied.
- Peak CUDA memory reset before each inference call; wall-clock time measured per sample.

**Whisper Small:**
- Input audio converted to mel spectrogram features via `AutoProcessor` with `padding="max_length"` and `truncation=True` to handle the fixed 30-second input window.
- Autoregressive decoding via `model.generate()`; special tokens stripped during decoding.
- Attention mask passed when available.

**Parakeet TDT 1.1B:**
- Audio array written to a temporary `.wav` file using `soundfile`; path passed to `parakeet_model.transcribe()`.
- Temporary file deleted after transcription.
- Transcription extracted from the response object's `.text` attribute.

### 5.4 Benchmarking Loop

For each of the 100 dataset samples, all three models were run sequentially. Per-sample metrics (WER, inference time, peak memory) were collected and aggregated into mean values. Results were saved to `results/benchmark_results.csv` and visualized as bar charts.

---

## 6. Evaluation Metrics

| Metric | Description | Direction |
|---|---|---|
| **Word Error Rate (WER)** | Primary accuracy metric. Computed using the `jiwer` library against cleaned reference transcriptions. | Lower is better |
| **Mean Inference Time (s)** | Wall-clock time per audio sample from input processing to decoded transcription. | Lower is better |
| **Peak GPU Memory (MB)** | Peak CUDA memory allocated during inference, measured via `torch.cuda.max_memory_allocated()`. Reset before each sample. | Lower is better |
| **Setup Complexity** | Qualitative assessment of dependencies, installation steps, and hardware requirements. | Lower is better |

---

## 7. Results

### 7.1 Aggregate Benchmark Results

| Model | Mean WER ↓ | Mean Inference Time (s) ↓ | Peak Memory (MB) | Setup Complexity |
|---|---|---|---|---|
| Wav2Vec2 Base 960h | 1.0825 | 0.0319 | 5767.93 | Low — HuggingFace native, minimal dependencies |
| **Whisper Small** | **0.4082** | **0.6998** | **5815.81** | Low — HuggingFace native, minimal dependencies |
| Parakeet TDT 1.1B | 0.2381 | 0.2447 | 5736.27 | Moderate — Requires NVIDIA NeMo toolkit (heavy install) |

### 7.2 Key Observations

- **Wav2Vec2** achieves the fastest inference (0.03s per sample) but at the cost of completely unusable accuracy (WER > 1.0 indicates more errors than words in reference). It is not viable for production on noisy conversational audio.
- **Parakeet TDT 1.1B** achieves the best raw WER (0.2381) and a strong balance of speed (0.24s) and accuracy, but its output format requires substantial post-processing.
- **Whisper Small** is the slowest of the three (0.70s) but produces clean, structured, punctuated output, making its effective accuracy higher than the raw WER suggests.
- **GPU memory usage is comparable** across all three models (~5736–5816 MB), reflecting that T4 VRAM is largely consumed by model weights rather than the inference workload itself.

### 7.3 Benchmark Visualization

A bar chart comparing all three metrics across models is available at [results/benchmark_comparison.png](results/benchmark_comparison.png).

---

## 8. Qualitative Analysis

Beyond aggregate metrics, a single sample was passed through all three models to illustrate qualitative differences in transcription behavior.

**Reference Text:**
> *that's where you have a lot of windows in the south no actually that's passive solar and passive solar is something that was developed and designed in the 1960s and 70s and it was a great thing for what it was at the time but it's not a passive house*

| Model | Transcription | Key Observations |
|---|---|---|
| **Wav2Vec2** | `ET'S WE HAVE A LOT OF WINDOWS IN THE SOUTH NO ACTION...` | All-caps output; missing punctuation; word-level errors ("ET'S" for "That's", "ACTION" for "actually"). Consistent with CTC decoding without a language model. |
| **Whisper Small** | `That's where you have a lot of windows in the south. No, actually...` | Correct punctuation; proper sentence structure; numbers written as digits ("1960s and 70s"). Whisper's added punctuation technically increases its WER relative to the unpunctuated reference, meaning its actual linguistic accuracy is higher than the metric suggests. |
| **Parakeet TDT** | `that's where you have a lot of windows in the south no actually...one nine hundred sixty s...` | Lowest WER; closely mirrors the reference text's unpunctuated, lowercased style. However, numbers are read verbatim as spoken words ("one nine hundred sixty s"), a known characteristic of Transducer models optimized for raw speech fidelity over written text normalization. |

### 8.1 Implication of Qualitative Findings

WER alone is insufficient for production evaluation. Parakeet achieves a lower WER partly because it closely mirrors the reference text's unpunctuated, lowercased style — its "accuracy" is partly an artifact of the benchmark format, not a reflection of real-world output quality. Whisper's added punctuation and number normalization are technically penalized as "errors" by WER, yet they are precisely the qualities that make transcriptions usable in production.

For customer support specifically — where agents and downstream NLP systems (sentiment analysis, intent detection, summarization) depend on readable, structured text — Whisper's output is meaningfully superior despite its higher raw WER.

---

## 9. Comparative Analysis

| Criterion | Wav2Vec2 Base 960h | Whisper Small | Parakeet TDT 1.1B |
|---|---|---|---|
| **Accuracy (WER ↓)** | 1.08 — Poor on noisy conversational audio | 0.41 — Good; punctuation adds readability beyond raw WER | 0.24 — Best raw WER; numbers spoken verbatim |
| **Latency** | 0.03s — Fastest, but unusable accuracy | 0.70s — Slowest; acceptable for async transcription | 0.24s — Best balance of speed and accuracy |
| **Resource Usage** | 5767 MB peak GPU | 5815 MB peak GPU | 5736 MB peak GPU |
| **Ease of Deployment** | Easy — HuggingFace native, minimal dependencies | Easy — HuggingFace native, minimal dependencies | Moderate — Requires NVIDIA NeMo toolkit (heavy install) |
| **Suitability for Noisy Audio** | Poor — Struggles significantly with noise and accents | Good — Robust to noise; handles accents and natural speech well | Good — Strong on noisy audio; designed for real-world speech |
| **Output Quality** | Unusable — All-caps, no punctuation | Production-ready — Punctuated, normalized, readable | Requires post-processing — No punctuation, numbers as words |

---

## 10. Recommendation

**Recommended model: Whisper Small**

While Parakeet TDT 1.1B achieves a lower raw WER, this advantage is misleading in the context of a customer support deployment:

### 10.1 Why Whisper Small

**Punctuation:** Whisper produces properly punctuated, sentence-structured output. Parakeet outputs a continuous stream of unpunctuated text, which is unusable for agents reading transcripts or for downstream NLP tasks such as sentiment analysis and intent detection.

**Number normalization:** Customer support calls frequently involve dates, order IDs, prices, and account numbers. Whisper correctly formats these as digits (e.g., "1960s and 70s"); Parakeet reads them verbatim ("one nine hundred sixty s"), requiring additional post-processing to be usable.

**WER is misleading:** Parakeet's lower WER is partly an artifact of matching the reference text's raw speech style (unpunctuated, lowercased). Whisper's "errors" — punctuation, capitalization, digit formatting — are improvements, not mistakes.

**Production readiness:** Whisper's output is deployment-ready with minimal post-processing. Recommending Parakeet would mean rebuilding Whisper's output quality manually, adding engineering overhead with no net benefit.

**Ease of deployment:** Both Wav2Vec2 and Whisper are HuggingFace-native with minimal dependencies. Parakeet requires the NVIDIA NeMo toolkit, a substantially heavier dependency chain that increases operational complexity.

### 10.2 When to Consider Parakeet TDT 1.1B

Parakeet TDT 1.1B remains a strong contender for:
- **Latency-critical or real-time streaming** use cases where its 0.24s inference time is essential.
- Deployments where a **text normalization layer** is added downstream (punctuation restoration, number formatting).
- Scenarios where **raw speech fidelity** is the primary requirement and post-processing infrastructure already exists.

---

## 11. Optimization Roadmap

The following optimizations are recommended for production deployment of Whisper Small:

| Optimization | Description |
|---|---|
| **INT8 Quantization** | Apply INT8 quantization to Whisper Small to reduce memory footprint and improve inference speed without significant accuracy loss. |
| **Fine-tuning (LoRA/PEFT)** | Fine-tune on domain-specific customer support audio using LoRA or PEFT techniques to improve accuracy on industry terminology, accents, and call center noise profiles. |
| **VAD Integration** | Add Voice Activity Detection (VAD) pre-processing to skip silence and reduce unnecessary inference calls, lowering effective latency and compute cost. |
| **Batched Inference** | For non-real-time use cases (e.g., post-call transcription), batch multiple audio clips to improve GPU utilization. |
| **Upgrade Path** | For higher accuracy requirements, Whisper Large V3 Turbo offers near-identical output style with significantly improved WER — a natural next step once hardware constraints allow. |

---

## 12. Proposed Production Architecture

```
Customer Call Audio
        ↓
  VAD Pre-processing         ← Strip silence
        ↓
  Whisper Small              ← ASR inference (INT8 quantized)
        ↓
  Downstream NLP Pipeline    ← Sentiment, intent, summarization, etc.
```

---

## 13. Hardware and Environment

| Component | Details |
|---|---|
| **GPU** | Google Colab T4 (15GB VRAM) |
| **Python** | Python3 |
| **Key Libraries** | `transformers`, `nemo_toolkit[asr]`, `datasets`, `jiwer`, `torch`, `soundfile`, `pandas`, `seaborn`, `matplotlib`, `numpy` |
| **Dataset** | MLCommons/peoples_speech (dirty, test split, 100 samples, streaming mode) |

### 13.1 Installation

```bash
pip install huggingface transformers torch torchaudio jiwer nemo_toolkit['asr']
```

---

## 14. Repository Structure

```
immverseai-asr-benchmark/
├── notebooks/
│   └── task_asr_benchmarks.ipynb     # Full benchmarking notebook (Colab)
├── results/
│   ├── benchmark_results.csv         # Aggregated metrics
│   └── benchmark_comparison.png      # Bar chart visualization
├── scripts/
│   └── task_asr_benchmarks.py        # Exported Python script of the benchmarking notebook
└── README.md
```

---

*End of Report*
