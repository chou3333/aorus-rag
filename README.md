# AORUS MASTER 16 AM6H RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) system for answering product specification questions about the **GIGABYTE AORUS MASTER 16 AM6H**.

The system is designed for resource-constrained environments and uses:

- Pure Python for chunking, retrieval, prompt construction, and evaluation
- `uv` for Python environment and dependency management
- `llama.cpp` through `llama-cpp-python` for local LLM inference
- Qwen2.5-1.5B-Instruct with Q4_K_M quantization
- Multilingual embeddings for Traditional Chinese and English queries
- NumPy-based vector similarity search
- Hybrid semantic + lexical retrieval
- Variant-aware context construction
- Deterministic structured responses for exact multi-variant comparisons
- Streaming response generation
- TTFT and TPS performance evaluation

No LangChain or LlamaIndex is used.

---

## 1. Project Goal

The goal of this project is to build a lightweight AI hardware assistant that can accurately answer specification questions about:

**GIGABYTE AORUS MASTER 16 AM6H**

Official product page:

https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp

The system supports:

- Traditional Chinese queries
- English queries
- Mixed Chinese-English queries
- Structured specification retrieval
- BZH / BYH / BXH variant-specific queries
- Streaming LLM responses
- Operation within a target 4 GB VRAM budget

Example questions:

```text
這台筆電的記憶體最高是多少？

What GPU does this laptop use?

這台 laptop 的 RAM 最大多少？

What GPU does the BYH model use?
```

---

## 2. System Architecture

The RAG pipeline is implemented without high-level RAG frameworks.

```text
GIGABYTE Product Specification Page
                |
                v
        Saved HTML Document
                |
                v
     Structured Key-Value Parsing
                |
                v
        Field-based Chunking
                |
                v
       Multilingual Embeddings
                |
                v
        NumPy Vector Index
                |
                v
 Semantic + Lexical Hybrid Retrieval
                |
                v
      Variant-aware Retrieval
                |
                v
 Context Consolidation / Comparison
                |
                v
   +-----------------------------+
   |                             |
   v                             v
LLM Generation        Deterministic Structured
via llama.cpp         Multi-Variant Comparison
   |                             |
   +-------------+---------------+
                 |
                 v
            Final Answer
```

The embedding model and vector retrieval run on CPU, while the LLM can be offloaded to GPU.

This allows the limited GPU memory budget to be reserved primarily for LLM generation.

For ordinary specification questions, generation is handled by the local LLM through `llama.cpp`.

For structured fields whose values differ across BZH, BYH, and BXH, the system can generate a deterministic comparison directly from the retrieved Key-Value records. This prevents the small language model from accidentally omitting one of the product variants.

---

## 3. Project Structure

```text
aorus-rag/
│
├── data/
│   ├── specs.json
│   └── chunks.json
│
├── embeddings/
│   └── embeddings.npy
│
├── models/
│   └── *.gguf
│       # ignored by Git
│
├── src/
│   └── aorus_rag/
│       ├── __init__.py
│       ├── scraper.py
│       ├── chunker.py
│       ├── embedder.py
│       ├── retriever.py
│       ├── generator.py
│       ├── rag.py
│       ├── evaluate_retrieval.py
│       └── evaluate_rag.py
│
├── tests/
│   ├── test_retrieval.py
│   └── test_specs.py
│
├── questions.json
├── rag_benchmark_results.csv
├── rag_benchmark_results_gpu.csv
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 4. Data Parsing

The AORUS MASTER 16 AM6H product page contains three variants:

- **AORUS MASTER 16 BZH**
- **AORUS MASTER 16 BYH**
- **AORUS MASTER 16 BXH**

The parser extracts specification records for all three models instead of stopping after the first model.

`specs.json` is organized by model name and specification category.

To rebuild the knowledge base after saving the product HTML to:

```text
data/aorus_master_16_am6h.html
```

run:

```bash
uv run python -m aorus_rag.scraper
uv run python -m aorus_rag.chunker
uv run python -m aorus_rag.embedder
```

The official product page contains structured fields such as:

```text
中央處理器
Intel Core Ultra 9 Processor 275HX

記憶體
Up to 64GB DDR5 5600MHz

電池
Li-ion 99Wh
```

The parser converts them into structured Key-Value data.

Example:

```json
{
  "中央處理器": [
    "Intel® Core™ Ultra 9 Processor 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)"
  ],
  "記憶體": [
    "Up to 64GB DDR5 5600MHz",
    "* 2x SO-DIMM sockets for expansion"
  ],
  "電池": [
    "Li-ion 99Wh"
  ]
}
```

### Variant Differences

An important difference between the three variants is the GPU configuration:

```text
BZH:
NVIDIA GeForce RTX 5090 Laptop GPU
24GB GDDR7
175W

BYH:
NVIDIA GeForce RTX 5080 Laptop GPU
16GB GDDR7
175W

BXH:
NVIDIA GeForce RTX 5070 Ti Laptop GPU
12GB GDDR7
140W
```

Many other specifications are shared across the three variants.

### HTML Acquisition

Direct HTTP requests to the official GIGABYTE product page returned HTTP `403 Access Denied` because of anti-bot protection.

Therefore, the raw product HTML was acquired through a normal browser session and parsed locally with BeautifulSoup.

The raw HTML file is excluded from the Git repository.

---

## 5. Chunking Strategy

Because the source data is a structured specification table rather than free-form text, this project uses **field-based chunking** instead of fixed-size character splitting.

Each specification category becomes one independent chunk.

Example:

```text
產品：AORUS MASTER 16 BZH
規格類別：記憶體

內容：
Up to 64GB DDR5 5600MHz
* 2x SO-DIMM sockets for expansion
```

This preserves the semantic relationship between each specification key and its value.

There are:

```text
3 product variants
×
17 specification categories
=
51 structured specification chunks
```

---

## 6. Embedding and Vector Index

Embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Each specification chunk is converted into a normalized 384-dimensional embedding.

The resulting embedding matrix has shape:

```text
(51, 384)
```

The embeddings are stored in:

```text
embeddings/embeddings.npy
```

The vector index is implemented directly with NumPy.

No external vector database is required.

Because embeddings are normalized, vector dot product is equivalent to cosine similarity:

```text
semantic_score = chunk_embedding · query_embedding
```

---

## 7. Retrieval

The system supports both general product questions and variant-specific questions.

Queries containing:

```text
BZH
BYH
BXH
```

are treated as variant-specific queries.

For example:

```text
BZH 使用什麼顯示卡？

What GPU does the BYH model use?

BXH 的 GPU 是什麼？
```

When the query explicitly specifies a variant, the answer path keeps only the retrieved specification from that product variant.

When no variant is specified, the RAG answer path retrieves the best matching specification chunk from each product variant.

For example:

```text
What GPU does this laptop use?
```

retrieves GPU information for:

```text
BZH
BYH
BXH
```

so that differences between variants can be preserved.

### Retrieval Process

For a user query, the system:

1. Converts the query into an embedding
2. Computes semantic similarity against all specification embeddings
3. Applies lightweight specification-category lexical boosting
4. Sorts the final scores
5. Selects the most relevant specification chunks
6. Applies variant-aware filtering or grouping

### Hybrid Retrieval

During initial testing, the multilingual embedding model handled pure Chinese and English queries correctly, but a mixed-language query produced a retrieval error.

Example:

```text
這台 laptop 的 RAM 最大多少？
```

Initial semantic-only ranking:

```text
Rank 1: 重量
Rank 2: 尺寸
Rank 3: 記憶體
```

To improve robustness for hardware terminology and code-switching queries, lightweight category aliases were added.

For example:

```text
RAM
memory
記憶體
```

are mapped to the same specification category.

The final ranking score becomes:

```text
final_score = semantic_score + lexical_bonus
```

After this improvement:

```text
Rank 1: 記憶體
```

This keeps the retrieval logic lightweight while improving multilingual hardware-query robustness.

---

## 8. LLM and Quantization

Generation model:

```text
Qwen2.5-1.5B-Instruct
```

Quantization:

```text
Q4_K_M
```

Model format:

```text
GGUF
```

Inference engine:

```text
llama.cpp
```

Python binding:

```text
llama-cpp-python
```

The GGUF model file is approximately:

```text
1.1 GB
```

The model itself is not committed to Git because of its size.

### Why This Model?

Qwen2.5-1.5B-Instruct was selected because:

- It is significantly smaller than typical 7B or 8B models
- It supports Chinese and English generation
- GGUF quantized versions are available
- Q4_K_M substantially reduces memory requirements
- It works well with `llama.cpp`
- It provides adequate performance for structured specification QA
- It leaves substantial memory headroom below the target 4 GB VRAM budget

---

## 9. Context Optimization

The system uses variant-aware context construction to reduce ambiguity for the small 1.5B language model.

### Explicit Variant Queries

When the query explicitly specifies:

```text
BZH
BYH
BXH
```

only the specification from that model is sent to the answer stage.

For example:

```text
What GPU does the BYH model use?
```

uses only the BYH GPU context.

### Shared Specifications

When no variant is specified, the system retrieves one matching specification from each product variant.

If the complete specification is identical across BZH, BYH, and BXH, duplicate contexts are collapsed into a single complete specification record.

For example, shared fields include:

```text
RAM
Battery
Weight
Communication
Ports
```

Importantly, the complete multi-line specification is preserved.

This avoids errors where only the first line of a specification is used.

For example, the communication field contains both Wi-Fi and Bluetooth information.

### Variant-Specific Specifications

If a specification differs across the three variants, each model's value is kept separately.

GPU is the main example:

```text
BZH: NVIDIA GeForce RTX 5090 Laptop GPU
BYH: NVIDIA GeForce RTX 5080 Laptop GPU
BXH: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

During evaluation, the 1.5B model occasionally omitted one of the variants even when all required information had been successfully retrieved.

Therefore, structured multi-variant comparisons use a deterministic answer path based directly on the retrieved Key-Value records.

This design:

- prevents variant omission
- reduces hallucination risk
- preserves exact model-to-specification relationships
- avoids requiring a larger language model

Ordinary natural-language questions still use LLM generation.

---

## 10. Streaming Generation

The system supports token streaming through `llama.cpp`.

Instead of waiting for the complete answer, generated content is displayed incrementally.

This enables measurement of:

- Time To First Token (TTFT)
- Generation Time
- Generated Token Count
- Tokens Per Second (TPS)

Generated token counts are calculated using the `llama.cpp` tokenizer rather than counting streaming callbacks.

Deterministic structured responses do not use LLM generation and therefore do not have TTFT or TPS measurements.

---

## 11. Installation

### Requirements

- Python 3.11
- `uv`
- Approximately 1.1 GB of storage for the GGUF model
- CPU-only operation is supported
- CUDA GPU is optional

Clone the repository:

```bash
git clone https://github.com/chou3333/aorus-rag.git
cd aorus-rag
```

Install dependencies:

```bash
uv sync
```

Check the Python version:

```bash
uv run python --version
```

Expected:

```text
Python 3.11.x
```

---

## 12. Download the GGUF Model

Download the model from Hugging Face:

```bash
uv run hf download \
Qwen/Qwen2.5-1.5B-Instruct-GGUF \
qwen2.5-1.5b-instruct-q4_k_m.gguf \
--local-dir models
```

Expected model path:

```text
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

---

## 13. Run the RAG System

### CPU Mode

By default:

```text
N_GPU_LAYERS=0
```

Run:

```bash
uv run python -m aorus_rag.rag
```

Example:

```text
請輸入問題：這台筆電的電池容量是多少？

=== Answer ===
電池容量為 99Wh。
```

---

## 14. GPU Mode

GPU offloading is controlled by:

```text
N_GPU_LAYERS
```

To fully offload supported model layers to GPU:

```bash
N_GPU_LAYERS=-1 PYTHONPATH=src .venv/bin/python -m aorus_rag.rag
```

### Google Colab CUDA Setup

The default locked `llama-cpp-python` package may be CPU-only.

For the Tesla T4 experiment, the CUDA-enabled wheel was installed into the project's `.venv`.

First remove the CPU build:

```bash
uv pip uninstall \
--python .venv/bin/python \
llama-cpp-python
```

Install the tested CUDA 12.5-compatible wheel:

```bash
uv pip install \
--python .venv/bin/python \
"https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu125/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl"
```

Verify GPU offloading:

```bash
.venv/bin/python -c \
"import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
```

Expected result:

```text
True
```

When using the manually installed CUDA build, use:

```bash
.venv/bin/python
```

or:

```bash
uv run --no-sync
```

to avoid dependency synchronization replacing the CUDA-enabled wheel with the default CPU build.

---

## 15. 4 GB VRAM Evaluation

GPU evaluation environment:

```text
GPU: NVIDIA Tesla T4
Physical VRAM: 15360 MiB
Model: Qwen2.5-1.5B-Instruct Q4_K_M
Context length: 2048
GPU offload: full (n_gpu_layers = -1)
Embedding model: CPU
Vector retrieval: CPU / NumPy
```

Measured GPU memory:

| Stage | GPU Memory |
|---|---:|
| Before model loading | 0 MiB |
| After model loading | 1397 MiB |
| After generation | 1423 MiB |

The highest observed usage was:

```text
1423 MiB
```

approximately:

```text
1.39 GiB
```

This is substantially below:

```text
4096 MiB
```

Therefore, the measured runtime GPU memory usage of the tested configuration remained below the target **4 GB VRAM budget**.

> Note: The experiment was executed on an NVIDIA Tesla T4 with substantially more than 4 GB of physical VRAM. The 4 GB compatibility assessment is based on measured GPU memory consumption of the tested configuration. It should not be interpreted as a claim that the system was executed on a physical 4 GB GPU.

---

## 16. Retrieval Benchmark

The retrieval benchmark contains **18 questions** covering:

- Traditional Chinese
- English
- Mixed Chinese-English queries
- CPU
- GPU
- RAM
- Battery
- Wi-Fi
- Bluetooth
- Thunderbolt
- Weight
- Size
- Color
- BZH variant-specific GPU query
- BYH variant-specific GPU query
- BXH variant-specific GPU query

Results:

| Metric | Result |
|---|---:|
| Total Questions | 18 |
| Top-1 Retrieval Accuracy | **100% (18/18)** |
| Top-3 Retrieval Accuracy | **100% (18/18)** |

The benchmark includes both general product questions and variant-specific questions.

The result should be interpreted as evidence that the structured retrieval strategy works reliably within the current AORUS MASTER 16 AM6H product domain rather than as a general-purpose RAG benchmark.

Run the retrieval benchmark with:

```bash
uv run python -m aorus_rag.evaluate_retrieval
```

---

## 17. End-to-End RAG Evaluation

The complete RAG pipeline was evaluated using an **18-question multilingual benchmark**.

Final answers were manually inspected against the parsed product specifications.

Overall result:

```text
18 / 18 answers correct
```

Of the 18 questions:

```text
16 questions used LLM generation
2 questions used deterministic structured comparison
```

The two deterministic questions are excluded from LLM TTFT and TPS averages because no language-model generation occurs for those responses.

### Mac CPU Baseline

Configuration:

```text
n_gpu_layers = 0
```

Results:

| Metric | Result |
|---|---:|
| Total Questions | 18 |
| LLM-generated Questions | 16 |
| Deterministic Questions | 2 |
| Correct Answers | **18/18** |
| Average LLM TTFT | **0.343 s** |
| Average LLM TPS | **68.78 tokens/s** |

The detailed CPU results are stored in:

```text
rag_benchmark_results.csv
```

### NVIDIA Tesla T4 GPU

Configuration:

```text
n_gpu_layers = -1
n_ctx = 2048
```

Results:

| Metric | Result |
|---|---:|
| Total Questions | 18 |
| LLM-generated Questions | 16 |
| Deterministic Questions | 2 |
| Correct Answers | **18/18** |
| Average LLM TTFT | **0.062 s** |
| Average LLM TPS | **126.21 tokens/s** |
| Peak observed GPU memory | **1423 MiB** |

The detailed GPU results are stored in:

```text
rag_benchmark_results_gpu.csv
```

GPU offloading significantly improved generation performance while keeping measured GPU memory usage well below the target 4 GB budget.

---

## 18. TTFT and TPS Definition

### TTFT

TTFT in the current benchmark measures:

```text
LLM request start
        |
        v
First generated content
```

Therefore, the reported value is **LLM-only TTFT**.

Retrieval is performed before the LLM timer begins.

This separation makes it possible to analyze LLM generation latency independently from retrieval latency.

### TPS

TPS represents generated tokens per second during the generation phase.

The token count is calculated with the `llama.cpp` tokenizer.

Only responses generated by the LLM are included in average TTFT and TPS calculations.

Deterministic structured responses are excluded because they do not invoke LLM generation.

---

## 19. Example Results

### Traditional Chinese

Question:

```text
這台筆電的記憶體最高是多少？
```

Answer:

```text
這台筆電的記憶體最高可達 64GB DDR5 5600MHz，
並且支援 2x SO-DIMM 插槽以進行擴充。
```

### English

Question:

```text
What GPU does this laptop use?
```

Answer:

```text
BZH: NVIDIA GeForce RTX 5090 Laptop GPU
BYH: NVIDIA GeForce RTX 5080 Laptop GPU
BXH: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### Mixed Language

Question:

```text
這台 laptop 的 RAM 最大多少？
```

Answer:

```text
這台 laptop 的 RAM 最大可支援至 64GB DDR5 5600MHz。
```

### Capability Question

Question:

```text
這台筆電有支援 Wi-Fi 7 嗎？
```

Answer:

```text
是的，這台筆電支援 Wi-Fi 7（802.11be 2x2）。
```

### Variant-Specific Question

Question:

```text
What GPU does the BYH model use?
```

Answer:

```text
NVIDIA GeForce RTX 5080 Laptop GPU
```

---

## 20. Evaluation Findings

### Finding 1: Structured chunking is effective

Because the source is a specification table, field-based chunking preserves meaningful Key-Value relationships better than arbitrary fixed-length splitting.

Each specification field becomes a separate retrieval unit, which makes exact product specification lookup straightforward.

### Finding 2: Multilingual embeddings alone are not always sufficient

Pure Chinese and English queries were generally handled correctly, but mixed-language hardware terminology could reduce semantic ranking quality.

Adding lightweight lexical aliases improved robustness for terms such as:

```text
RAM
memory
記憶體
```

### Finding 3: Variant-aware context improves small-model reliability

The product page contains three variants whose specifications are mostly shared but differ in important fields such as GPU configuration.

For identical specifications, duplicate contexts are consolidated into one complete record.

For variant-specific differences, each model's value is preserved separately.

A deterministic structured-answer path is used for multi-variant comparisons because the 1.5B model occasionally omitted variants even when retrieval was correct.

This improved reliability without requiring a larger language model.

### Finding 4: Preserving full multi-line fields is important

Some product fields contain multiple lines.

For example, communication specifications contain both Wi-Fi and Bluetooth information, while port specifications contain Thunderbolt details.

Using only the first line of these fields caused incorrect answers during development.

The final system therefore preserves the complete specification record when variants share the same field.

### Finding 5: Quantization provides substantial memory savings

The Q4_K_M GGUF model required approximately 1.4 GB of GPU memory during the tested workload.

Peak observed GPU memory was:

```text
1423 MiB
```

which leaves substantial headroom below the 4 GB target.

### Finding 6: GPU offloading improves generation throughput

Average LLM generation throughput increased from:

```text
68.78 tokens/s
```

on the Mac CPU baseline to:

```text
126.21 tokens/s
```

on the Tesla T4 GPU test.

Average LLM TTFT decreased from:

```text
0.343 s
```

to:

```text
0.062 s
```

---

## 21. Limitations

This project has several limitations.

First, the knowledge base currently covers one product family and three variants:

```text
BZH
BYH
BXH
```

Second, the evaluation set contains only 18 direct specification questions, so the reported accuracy should not be interpreted as general-purpose RAG performance.

Third, category aliases are manually defined for this limited hardware domain.

Fourth, the current TTFT metric measures LLM generation latency and does not include retrieval latency.

Fifth, two multi-variant comparison questions use deterministic structured responses and therefore do not contribute to the LLM TTFT or TPS averages.

Sixth, the 4 GB compatibility evaluation was performed by measuring GPU memory consumption on a Tesla T4 rather than by executing the system on a physical 4 GB GPU.

Finally, the project currently relies on manually acquired HTML because direct automated requests to the product page were blocked by anti-bot protection.

---

## 22. Possible Future Improvements

Future work could include:

- Measuring retrieval latency separately
- Reporting complete end-to-end latency
- Expanding the benchmark with paraphrased questions
- Adding adversarial retrieval questions
- Adding unsupported-information questions to evaluate hallucination resistance
- Comparing multiple embedding models
- Comparing 1.5B and 3B quantized language models
- Implementing automatic answer correctness evaluation
- Extending the knowledge base to additional GIGABYTE products
- Adding dynamic HTML ingestion
- Evaluating different context lengths
- Evaluating different quantization levels
- Testing on an actual 4 GB GPU
- Comparing CPU, partial GPU offload, and full GPU offload

---

## 23. Main Technologies

```text
Python 3.11

uv

BeautifulSoup

NumPy

Sentence Transformers

paraphrase-multilingual-MiniLM-L12-v2

Qwen2.5-1.5B-Instruct

GGUF Q4_K_M

llama.cpp

llama-cpp-python
```

---

## 24. Summary

This project demonstrates a lightweight RAG implementation designed for resource-constrained consumer hardware.

Key results:

```text
No LangChain / LlamaIndex

Pure Python RAG core

3 product variants:
BZH / BYH / BXH

51 structured specification chunks

Traditional Chinese + English + mixed-language support

Hybrid semantic + lexical retrieval

Variant-aware retrieval and context construction

Top-1 retrieval accuracy:
100% (18/18)

Top-3 retrieval accuracy:
100% (18/18)

End-to-end benchmark:
18/18 answers correct

16 LLM-generated responses

2 deterministic structured responses

Streaming LLM generation

Mac CPU average LLM TTFT:
0.343 s

Mac CPU average LLM TPS:
68.78 tokens/s

Tesla T4 average LLM TTFT:
0.062 s

Tesla T4 average LLM TPS:
126.21 tokens/s

Peak observed GPU memory:
1423 MiB

Target VRAM budget:
4096 MiB
```

The results show that a small quantized language model combined with structured retrieval, variant-aware context construction, and deterministic handling of exact multi-variant comparisons can provide accurate product-specification question answering while keeping measured GPU memory usage well below the target 4 GB budget.