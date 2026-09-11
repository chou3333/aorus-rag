# AORUS MASTER 16 AM6H — Local Product Specification RAG Assistant

A lightweight RAG system for answering product-specification questions about the GIGABYTE AORUS MASTER 16 AM6H.

The project is implemented in pure Python without LangChain or LlamaIndex. It supports Traditional Chinese, English, and mixed-language queries for the BZH, BYH, and BXH variants.

Source: [GIGABYTE AORUS MASTER 16 AM6H Specifications](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)

---

## 1. Features

- Structured parsing of official product specifications
- Variant-aware retrieval for BZH / BYH / BXH
- 51 chunks: 3 variants × 17 specification fields
- Multilingual vector retrieval
- Traditional Chinese / English / mixed-language queries
- Streaming LLM generation
- Typo normalization and ambiguous-query handling
- Structured answers for comparisons and multi-field questions
- TTFT and TPS measurement
- Robustness benchmark and VRAM monitoring

No LangChain or LlamaIndex is used.

---

## 2. Architecture

```text
Official HTML
    ↓
Structured Key-Value Parsing
    ↓
Chunking by Variant + Specification Field
    ↓
Multilingual Embeddings
    ↓
NumPy Vector Index
    ↓
Query Normalization / Scope Check
    ↓
Vector Retrieval
    ↓
Answer Routing
    ├── Structured Python Answer
    └── llama.cpp Streaming Generation
```

### Retrieval

Embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Embeddings and retrieval run on CPU.

Normalized vectors are searched using NumPy dot products, which correspond to cosine similarity.

### Generation

Generation model:

```text
Qwen2.5-1.5B-Instruct-GGUF
```

Quantization:

```text
Q4_K_M
```

Inference engine:

```text
llama-cpp-python
```

The LLM uses streaming generation with `stream=True`.

For exact comparisons, multi-field questions, and product overviews, the system can return structured source-based answers directly instead of relying on the LLM. This helps reduce omitted variants and hallucinated specification values.

---

## 3. Quick Start

Requirements:

- Python 3.11
- `uv`

Install dependencies:

```bash
uv python install 3.11
uv sync --locked --python 3.11
```

Download the GGUF model:

```bash
uv run hf download \
  Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --local-dir models
```

Build the embedding index if needed:

```bash
uv run python -m aorus_rag.embedder
```

Run the assistant:

```bash
uv run python -m aorus_rag.rag
```

Example questions:

```text
What GPU does the BYH model use?

BYH 和 BXH 的 GPU 差異？

這台 laptop 的 RAM 最大多少？

記意體最大多少？

AORUS MASTER 16 BZH介紹一下

電池可以撐幾小時？
```

---

## 4. 4GB VRAM Design

The project targets a 4GB VRAM environment.

Resource-control choices include:

| Component | Setting |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Quantization | Q4_K_M |
| GGUF size | ~1.1 GB |
| Context length | 2048 |
| Output limit | 256 tokens |
| Batch size | 128 |
| Embeddings | CPU |
| Retrieval | CPU / NumPy |

Keeping embedding and retrieval on CPU reserves GPU memory for LLM generation.

### GPU Validation

The current revision was tested in Google Colab on an NVIDIA Tesla T4 with GPU offloading enabled.

The full robustness benchmark was monitored with `nvidia-smi` every 100 ms.

Results:

| Metric | Result |
|---|---:|
| Regression checks | 45 / 45 |
| LLM-generated cases | 23 |
| Mean LLM TTFT | ~0.035 s |
| Mean estimated TPS | ~125.5 tokens/s |
| VRAM samples | 182 |
| Observed peak GPU memory | 1205 MiB |

The observed peak of **1205 MiB** is below the **4 GiB target**.

The benchmark was performed on a Tesla T4, which has more than 4GB of physical VRAM. Therefore, this result demonstrates observed usage below the 4GB budget rather than execution on a physical 4GB GPU.

---

## 5. Evaluation

Run automated tests:

```bash
uv run pytest -q
```

Current result:

```text
44 passed
```

Run retrieval evaluation:

```bash
uv run python -m aorus_rag.evaluate_retrieval
```

Result:

```text
Top-1 Accuracy: 18/18
Top-3 Accuracy: 18/18
```

Run the end-to-end robustness benchmark:

```bash
uv run python -m aorus_rag.evaluate_robustness
```

Result:

```text
Regression checks: 45/45
```

The robustness benchmark includes cases covering:

- variant isolation
- Chinese and English typos
- multi-field queries
- variant comparisons
- ambiguous questions
- unsupported information
- battery-runtime hallucination prevention
- installed vs maximum RAM capacity
- prompt-injection-like requests

`45/45` represents successful regression checks and should not be interpreted as 100% accuracy on arbitrary unseen questions.

---

## 6. Example Robustness Cases

### Typo

```text
Q: 記意體最大多少？
A: 記憶體最大支援 64GB DDR5 5600MHz。
```

### Variant-specific question

```text
Q: What GPU does the BYH model use?
A: The BYH model uses the NVIDIA GeForce RTX 5080 Laptop GPU.
```

### Unsupported measurement

```text
Q: 電池可以撐幾小時？
```

The official specification only provides a 99Wh battery capacity and does not provide measured battery runtime, so the system does not invent an estimated number of hours.

### Invalid specification premise

```text
Q: BYH 的記憶體是不是 128GB？
```

The system returns the official maximum specification of 64GB instead of accepting the incorrect premise.

---

## 7. Project Structure

```text
aorus-rag/
├── data/
│   ├── specs.json
│   └── chunks.json
├── embeddings/
├── models/
├── src/
│   └── aorus_rag/
│       ├── chunker.py
│       ├── embedder.py
│       ├── evaluate_rag.py
│       ├── evaluate_retrieval.py
│       ├── evaluate_robustness.py
│       ├── query.py
│       ├── rag.py
│       ├── retriever.py
│       └── scraper.py
├── tests/
├── questions.json
├── robustness_results.json
├── robustness_results_colab.json
├── gpu_memory.csv
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 8. Known Limitations

- The knowledge base only covers the AORUS MASTER 16 AM6H product family.
- Pricing and availability are not live.
- Battery-runtime and game-FPS measurements are not included in the official specification data.
- Typo and scope handling are rule-based and cannot cover every possible query.
- The GPU benchmark was performed on a Tesla T4 rather than a physical 4GB GPU.
- GPU memory was sampled every 100 ms, so extremely short peaks may not be captured.