# AORUS MASTER 16 AM6H RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) system for answering product specification questions about the **GIGABYTE AORUS MASTER 16 AM6H**.

The system is designed for resource-constrained environments and uses:

- Pure Python for chunking, retrieval, prompt construction, and evaluation
- `uv` for Python environment and dependency management
- `llama.cpp` through `llama-cpp-python` for local LLM inference
- Qwen2.5-1.5B-Instruct with Q4_K_M quantization
- Multilingual embeddings for Traditional Chinese and English queries
- NumPy-based vector similarity search
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
- Streaming LLM responses
- Operation within a 4 GB VRAM budget

Example questions:

```text
這台筆電的記憶體最高是多少？

What GPU does this laptop use?

這台 laptop 的 RAM 最大多少？
```

---

## 2. System Architecture

The RAG pipeline is implemented without high-level RAG frameworks.

```text
GIGABYTE Product Specification Page
                |
                v
        HTML Specification Data
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
          Top-1 Context
                |
                v
        Prompt Construction
                |
                v
 Qwen2.5-1.5B-Instruct Q4_K_M
          via llama.cpp
                |
                v
       Streaming Response
```

The embedding model and vector retrieval run on CPU, while the LLM can be offloaded to GPU.

This allows the limited GPU memory budget to be reserved primarily for generation.

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
│   └── *.gguf                 # ignored by Git
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
├── questions.json
├── rag_benchmark_results.csv
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 4. Data Parsing

The official product page contains structured specification fields such as:

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
產品：AORUS MASTER 16 AM6H
規格類別：記憶體
內容：
Up to 64GB DDR5 5600MHz
* 2x SO-DIMM sockets for expansion
```

This preserves the semantic relationship between each specification key and its value.

A total of **17 specification chunks** are generated.

---

## 6. Embedding and Vector Index

Embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Each specification chunk is converted into a normalized 384-dimensional embedding.

The resulting embedding matrix has shape:

```text
(17, 384)
```

The embeddings are stored in:

```text
embeddings/embeddings.npy
```

The vector index is implemented directly with NumPy.

No external vector database is required.

---

## 7. Retrieval

For a user query, the system:

1. Converts the query into an embedding
2. Computes semantic similarity against all specification embeddings
3. Applies lightweight specification-category lexical boosting
4. Sorts the final scores
5. Returns the most relevant specification chunk

Because the embeddings are normalized, vector dot product can be used for cosine similarity.

Conceptually:

```text
semantic_score = chunk_embedding · query_embedding
```

### Hybrid Retrieval

During initial testing, the multilingual embedding model handled pure Chinese and pure English queries correctly, but a mixed-language query produced a retrieval error.

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

The GGUF model file is approximately **1.1 GB**.

The model itself is not committed to Git because of its size.

### Why This Model?

Qwen2.5-1.5B-Instruct was selected because:

- It is significantly smaller than typical 7B or 8B models
- It supports Chinese and English generation
- GGUF quantized versions are available
- Q4_K_M significantly reduces memory requirements
- It works well with `llama.cpp`
- It leaves substantial memory headroom under the target 4 GB VRAM budget

---

## 9. Context Optimization

Initially, the generator received the Top-3 retrieved chunks.

Retrieval accuracy was high, but the 1.5B model occasionally failed to extract information that was already present in the retrieved context.

Examples included:

```text
Wi-Fi 7
Thunderbolt 5
Dark Tide color
```

Since retrieval evaluation showed that the correct specification was consistently ranked first, generation was changed from:

```text
Top-3 Context
```

to:

```text
Top-1 Context
```

The context format was also simplified to:

```text
Product
Specification Name
Specification Value
```

This reduced irrelevant context and improved answer reliability for the small language model.

---

## 10. Streaming Generation

The system supports token streaming through `llama.cpp`.

Instead of waiting for the entire answer to finish, generated content is displayed incrementally.

This also enables measurement of:

- Time To First Token (TTFT)
- Generation Time
- Generated Token Count
- Tokens Per Second (TPS)

Generated token counts are calculated using the `llama.cpp` tokenizer rather than counting streaming callbacks.

---

## 11. Installation

### Requirements

- Python 3.11
- `uv`
- Approximately 1.1 GB storage for the GGUF model
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

The expected model path is:

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
這台筆電的電池容量為 99Wh。
```

---

## 14. GPU Mode

GPU offloading can be controlled with the environment variable:

```text
N_GPU_LAYERS
```

To fully offload supported model layers to GPU:

```bash
N_GPU_LAYERS=-1 PYTHONPATH=src .venv/bin/python -m aorus_rag.rag
```

In the Google Colab GPU experiment, a CUDA-enabled build of `llama-cpp-python` was used.

---

## 15. 4 GB VRAM Evaluation

GPU evaluation environment:

```text
GPU: NVIDIA Tesla T4
Available VRAM: 15360 MiB
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

The highest observed usage during this test was approximately:

```text
1423 MiB
```

This is substantially below the target:

```text
4096 MiB
```

Therefore, the tested configuration fits within the required **4 GB VRAM budget**.

> Note: The experiment was executed on a Tesla T4 with more than 4 GB physical VRAM. The compatibility claim is based on measured memory consumption of the tested RAG generation configuration, rather than claiming that the experiment was executed on a physical 4 GB GPU.

---

## 16. Retrieval Benchmark

The retrieval benchmark contains **15 questions** across:

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

Results:

| Metric | Result |
|---|---:|
| Total Questions | 15 |
| Top-1 Retrieval Accuracy | **100% (15/15)** |
| Top-3 Retrieval Accuracy | **100% (15/15)** |

This benchmark focuses on direct product specification lookup.

Therefore, the result should be interpreted as evidence that the structured retrieval strategy works reliably for the current product domain rather than as a general-purpose RAG benchmark.

Run the retrieval benchmark with:

```bash
uv run python -m aorus_rag.evaluate_retrieval
```

---

## 17. End-to-End RAG Evaluation

The complete RAG pipeline was evaluated using the same 15-question multilingual benchmark.

Final answers were manually inspected against the parsed product specifications.

Result:

```text
15 / 15 answers correct
```

### Mac CPU Baseline

Configuration:

```text
n_gpu_layers = 0
```

Results:

| Metric | Result |
|---|---:|
| Questions | 15 |
| Average LLM TTFT | 0.179 s |
| Average TPS | 68.58 tokens/s |

### NVIDIA Tesla T4 GPU

Configuration:

```text
n_gpu_layers = -1
n_ctx = 2048
```

Results:

| Metric | Result |
|---|---:|
| Questions | 15 |
| Average LLM TTFT | **0.035 s** |
| Average TPS | **128.52 tokens/s** |
| Peak observed GPU memory | **1423 MiB** |

The GPU configuration significantly improved generation throughput while remaining well below the 4 GB VRAM target.

---

## 18. TTFT Definition

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

---

## 19. Example Results

### Traditional Chinese

Question:

```text
這台筆電的記憶體最高是多少？
```

Answer:

```text
這台筆電的記憶體最高可支援至 64GB DDR5 5600MHz。
```

### English

Question:

```text
What GPU does this laptop use?
```

Answer:

```text
NVIDIA GeForce RTX 5090 Laptop GPU
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
支援，規格值為 WIFI 7 (802.11be 2x2)。
```

---

## 20. Evaluation Findings

### Finding 1: Structured chunking is effective

Because the source is a specification table, field-based chunking preserves meaningful Key-Value relationships better than arbitrary fixed-length splitting.

### Finding 2: Multilingual embeddings alone are not always sufficient

Pure Chinese and English queries were handled correctly, but mixed-language hardware terminology could reduce semantic ranking quality.

Adding lightweight lexical aliases improved robustness.

### Finding 3: Smaller context improved small-model reliability

Although Top-3 retrieval provides more context, the 1.5B model occasionally became less reliable when irrelevant chunks were included.

Using the highest-confidence Top-1 result reduced context noise and improved final answer quality.

### Finding 4: Quantization provides substantial memory savings

The Q4_K_M GGUF model required approximately 1.4 GB GPU memory during the tested generation workload, leaving significant headroom below the 4 GB target.

### Finding 5: GPU offloading improves generation throughput

The observed average generation rate increased from:

```text
68.58 tokens/s
```

on the Mac CPU baseline to:

```text
128.52 tokens/s
```

on the Tesla T4 GPU test.

---

## 21. Limitations

This project has several limitations.

First, the knowledge base currently contains only one product specification page.

Second, the evaluation set contains only 15 direct specification questions, so the reported accuracy does not represent general-purpose RAG performance.

Third, category aliases are manually defined for this limited hardware domain.

Fourth, the current TTFT metric measures LLM generation latency and does not include retrieval latency.

Finally, the 4 GB compatibility evaluation was performed by measuring GPU memory usage on a Tesla T4 rather than using a physical 4 GB GPU.

---

## 22. Possible Future Improvements

Future work could include:

- Measuring retrieval latency separately
- Reporting complete end-to-end TTFT
- Expanding the benchmark with paraphrased and adversarial questions
- Adding unsupported-information questions to evaluate hallucination resistance
- Comparing multiple embedding models
- Comparing 1.5B and 3B quantized language models
- Implementing automatic answer correctness evaluation
- Extending the knowledge base to multiple GIGABYTE products
- Adding dynamic HTML ingestion
- Evaluating different context lengths and GPU offloading configurations

---

## 23. Main Technologies

```text
Python 3.11
uv
BeautifulSoup
NumPy
Sentence Transformers
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
17 structured specification chunks
Traditional Chinese + English support
Hybrid semantic + lexical retrieval
Top-1 retrieval accuracy: 100%
Top-3 retrieval accuracy: 100%
15/15 benchmark answers correct
Streaming generation
Tesla T4 average TTFT: 0.035 s
Tesla T4 average TPS: 128.52 tokens/s
Peak observed GPU memory: 1423 MiB
Target VRAM budget: 4096 MiB
```

The results show that a small quantized language model combined with structured retrieval can provide accurate product-specification question answering while keeping GPU memory usage well below the 4 GB target.