import os
import time
from llama_cpp import Llama

from aorus_rag.retriever import retrieve


MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))


print("正在載入 LLM...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)
print(f"GPU layers: {N_GPU_LAYERS}")
print("LLM 載入完成")


def build_context(results):
    chunk = results[0]["chunk"]

    return (
        f"產品：{chunk['product']}\n"
        f"規格名稱：{chunk['category']}\n"
        f"規格值：\n{chunk['content']}"
    )


def answer_question(query):
    results = retrieve(query, top_k=1)

    context = build_context(results)
    system_prompt = """
    你是一個 GIGABYTE AORUS MASTER 16 AM6H 產品規格助理。

    只能根據提供的產品規格回答。

    規則：
    1. 「規格值」中的內容就是答案依據。
    2. 如果使用者問是否支援某功能，只要該功能或版本出現在「規格值」中，就回答支援，並附上該規格。
    3. 比對名稱時忽略大小寫、空格、連字號、™、® 等符號差異。
    4. 只有當「規格值」完全沒有相關資訊時，才能回答找不到資訊。
    5. 中文問題用繁體中文回答；英文問題用英文回答。
    6. 回答簡潔但要包含實際規格值。
    """

    user_prompt = f"""
Context:
{context}

Question:
{query}
"""

    start_time = time.perf_counter()
    first_token_time = None

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=200,
        temperature=0.1,
        stream=True,
    )

    print()
    print("=== Answer ===")

    answer_parts = []

    for chunk in response:
        delta = chunk["choices"][0]["delta"]

        if "content" in delta:
            text = delta["content"]

            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                print(text, end="", flush=True)

                answer_parts.append(text)

    print()

    answer = "".join(answer_parts)
    end_time = time.perf_counter()

    generated_tokens = llm.tokenize(
        answer.encode("utf-8"),
        add_bos=False
    )

    token_count = len(generated_tokens)

    if first_token_time is not None:
        ttft = first_token_time - start_time
        generation_time = end_time - first_token_time

        if generation_time > 0:
            tps = token_count / generation_time
        else:
            tps = 0
    else:
        ttft = 0
        generation_time = 0
        tps = 0

    print()
    print("=== Performance ===")
    print(f"TTFT: {ttft:.3f} seconds")
    print(f"Generated Tokens: {token_count}")
    print(f"Generation Time: {generation_time:.3f} seconds")
    print(f"TPS: {tps:.2f} tokens/second")
    metrics = {
        "ttft": ttft,
        "tps": tps,
        "generated_tokens": token_count,
        "generation_time": generation_time,
    }

    return answer, results, metrics
if __name__ == "__main__":
    query = input("請輸入問題：")

    answer, results, metrics = answer_question(query)
    print()
    print("=== Retrieval Results ===")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['chunk']['category']} "
            f"(score={result['final_score']:.4f})"
        )