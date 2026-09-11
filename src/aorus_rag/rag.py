import os
import re
import time
from llama_cpp import Llama

from aorus_rag.retriever import retrieve


MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))
CATEGORY_ALIASES = {
    "中央處理器": [
        "cpu",
        "processor",
        "處理器",
        "中央處理器",
    ],
    "顯示晶片": [
        "gpu",
        "graphics",
        "顯示晶片",
        "顯示卡",
    ],
    "顯示器": [
        "display",
        "screen",
        "monitor",
        "螢幕",
        "顯示器",
        "解析度",
        "refresh rate",
        "更新率",
        "刷新率",
    ],
    "記憶體": [
        "ram",
        "memory",
        "記憶體",
    ],
    "儲存裝置": [
        "storage",
        "ssd",
        "硬碟",
        "儲存",
    ],
    "連接埠": [
        "port",
        "ports",
        "usb",
        "hdmi",
        "thunderbolt",
        "連接埠",
        "接口",
    ],
    "音效": [
        "audio",
        "speaker",
        "speakers",
        "sound",
        "音效",
        "喇叭",
    ],
    "通訊": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "lan",
        "network",
        "網路",
        "藍牙",
        "通訊",
    ],
    "視訊鏡頭": [
        "webcam",
        "camera",
        "鏡頭",
        "視訊鏡頭",
    ],
    "安全裝置": [
        "tpm",
        "security",
        "安全",
        "安全裝置",
    ],
    "電池": [
        "battery",
        "電池",
    ],
    "變壓器": [
        "adapter",
        "charger",
        "power adapter",
        "充電器",
        "變壓器",
    ],
    "尺寸": [
        "size",
        "dimension",
        "dimensions",
        "尺寸",
    ],
    "重量": [
        "weight",
        "重量",
        "多重",
    ],
    "顏色": [
        "color",
        "colour",
        "顏色",
    ],
}

print("正在載入 LLM...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)

print(f"GPU layers: {N_GPU_LAYERS}")
print("LLM 載入完成")

def detect_categories(query):
    query_lower = query.lower()

    matched_categories = []

    for category, aliases in CATEGORY_ALIASES.items():

        for alias in aliases:

            alias_lower = alias.lower()

            if re.fullmatch(r"[a-zA-Z0-9 -]+", alias_lower):
                pattern = (
                    r"(?<![a-zA-Z0-9])"
                    + re.escape(alias_lower)
                    + r"(?![a-zA-Z0-9])"
                )

                if re.search(pattern, query_lower):
                    matched_categories.append(category)
                    break

            else:
                if alias_lower in query_lower:
                    matched_categories.append(category)
                    break

    return matched_categories
def query_has_variant(query):
    query_upper = query.upper()

    return any(
        variant in query_upper
        for variant in ["BZH", "BYH", "BXH"]
    )


def get_target_variant(query):
    query_upper = query.upper()

    for variant in ["BZH", "BYH", "BXH"]:
        if variant in query_upper:
            return variant

    return None


def get_main_value(chunk):
    lines = [
        line.strip()
        for line in chunk["content"].splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    return lines[0]


def get_response_language(query):
    """
    如果問題包含中文字元，就使用繁體中文。
    否則使用英文。
    """
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh"

    return "en"


def build_context(results, compare_variants=False):

    # 多型號情況
    if compare_variants and len(results) > 1:

        categories = {
            result["chunk"]["category"]
            for result in results
        }

        # 三筆都是同一規格類別
        if len(categories) == 1:

            contents = [
                result["chunk"]["content"].strip()
                for result in results
            ]

            # ==================================================
            # 情況 1：
            # 三個 variant 的完整規格完全相同
            #
            # 例如：
            # - 通訊
            # - 連接埠
            # - 重量
            # - 電池
            #
            # 這時候只需要提供一份完整規格，
            # 但不能只取第一行。
            # ==================================================
            if len(set(contents)) == 1:

                chunk = results[0]["chunk"]

                return (
                    "此規格適用於 BZH、BYH、BXH 三個型號。\n"
                    f"規格名稱：{chunk['category']}\n"
                    f"規格值：\n{chunk['content']}"
                )

            # ==================================================
            # 情況 2：
            # 三個 variant 的規格不同
            #
            # 例如 GPU。
            #
            # 為避免小模型被過多資訊干擾，
            # 每個型號只保留主要規格值。
            # ==================================================
            context_parts = []

            for result in results:
                chunk = result["chunk"]

                main_value = get_main_value(chunk)

                context_parts.append(
                    f"型號：{chunk['product']}\n"
                    f"規格：{chunk['category']}\n"
                    f"主要規格值：{main_value}"
                )

            return "\n\n".join(context_parts)

    # 一般情況保留完整規格
    return "\n\n".join(
        f"產品：{chunk['product']}\n"
        f"規格名稱：{chunk['category']}\n"
        f"規格值：\n{chunk['content']}"
        for chunk in (
            result["chunk"]
            for result in results
        )
    )


def build_forced_comparison_answer(results, query):
    """
    如果三個型號是同一 category，
    但規格內容不同，
    建立三個型號的比較答案。

    - 一般 GPU 問題：只列主要 GPU 型號
    - 詳細規格問題：列出完整規格內容
    """

    if len(results) <= 1:
        return None

    categories = {
        result["chunk"]["category"]
        for result in results
    }

    if len(categories) != 1:
        return None

    full_contents = [
        result["chunk"]["content"].strip()
        for result in results
    ]

    # 三個型號內容完全一樣，不需要做 variant comparison
    if len(set(full_contents)) == 1:
        return None

    query_lower = query.lower()

    # 判斷使用者是不是在問「完整顯示晶片規格」
    wants_full_detail = any(
        keyword in query_lower
        for keyword in [
            "顯示晶片",
            "顯示卡規格",
            "gpu 規格",
            "gpu spec",
            "gpu specs",
            "graphics spec",
            "graphics specs",
            "graphics specification",
        ]
    )

    lines = []

    for result in results:
        chunk = result["chunk"]
        variant = chunk["product"].split()[-1]

        if wants_full_detail:
            # 保留完整多行規格
            value = chunk["content"].strip()

            lines.append(
                f"{variant}:\n{value}"
            )

        else:
            # 一般 GPU 問題只取第一行
            main_value = get_main_value(chunk)

            lines.append(
                f"{variant}: {main_value}"
            )

    return "\n\n".join(lines)
def answer_question(query):

    # ==================================================
    # 1. 先偵測使用者是不是一次問多個規格類別
    # ==================================================
    detected_categories = detect_categories(query)

    if len(detected_categories) > 1:
        combined_answers = []
        combined_results = []
        response_language = get_response_language(query)

        category_labels = {
            "zh": {
                "中央處理器": "中央處理器",
                "顯示晶片": "顯示晶片",
                "顯示器": "顯示器",
                "記憶體": "記憶體",
                "儲存裝置": "儲存裝置",
                "連接埠": "連接埠",
                "音效": "音效",
                "通訊": "通訊",
                "視訊鏡頭": "視訊鏡頭",
                "安全裝置": "安全裝置",
                "電池": "電池",
                "變壓器": "變壓器",
                "尺寸": "尺寸",
                "重量": "重量",
                "顏色": "顏色",
            },
            "en": {
                "中央處理器": "CPU",
                "顯示晶片": "GPU",
                "顯示器": "Display",
                "記憶體": "Memory",
                "儲存裝置": "Storage",
                "連接埠": "Ports",
                "音效": "Audio",
                "通訊": "Connectivity",
                "視訊鏡頭": "Webcam",
                "安全裝置": "Security",
                "電池": "Battery",
                "變壓器": "Power Adapter",
                "尺寸": "Dimensions",
                "重量": "Weight",
                "顏色": "Color",
            },
        }

        for category in detected_categories:

            # 用 category 名稱本身做 retrieval
            category_results = retrieve(
                category,
                top_k=1,
                per_product=True,
            )

            # 只保留真正屬於這個 category 的結果
            category_results = [
                result
                for result in category_results
                if result["chunk"]["category"] == category
            ]

            if not category_results:
                continue

            combined_results.extend(category_results)

            contents = [
                result["chunk"]["content"].strip()
                for result in category_results
            ]

            # ==================================================
            # 三個 variant 的完整內容都相同
            # 例如：
            # 重量 / 尺寸 / 電池 / RAM
            # ==================================================
            label = category_labels[response_language].get(
                category,
                category,
            )

            if len(set(contents)) == 1:

                value = category_results[0]["chunk"]["content"].strip()

                combined_answers.append(
                    f"{label}:\n{value}"
                )

            # ==================================================
            # 三個 variant 不同
            # 例如 GPU
            # ==================================================
            else:
                variant_lines = []

                for result in category_results:
                    chunk = result["chunk"]
                    variant = chunk["product"].split()[-1]

                    variant_lines.append(
                        f"{variant}:\n{chunk['content'].strip()}"
                    )

                combined_answers.append(
                    f"{label}:\n"
                    + "\n\n".join(variant_lines)
                )

        if combined_answers:
            answer = "\n\n".join(combined_answers)

            print()
            print("=== Answer ===")
            print(answer)

            generated_tokens = llm.tokenize(
                answer.encode("utf-8"),
                add_bos=False,
            )

            token_count = len(generated_tokens)

            metrics = {
                "ttft": 0.0,
                "tps": 0.0,
                "generated_tokens": token_count,
                "generation_time": 0.0,
            }

            print()
            print("=== Performance ===")
            print("TTFT: N/A (deterministic multi-category response)")
            print(f"Generated Tokens: {token_count}")
            print("Generation Time: N/A")
            print("TPS: N/A")

            return answer, combined_results, metrics

    # ==================================================
    # 2. 如果不是 multi-category，就走原本流程
    # ==================================================

    has_variant = query_has_variant(query)

    results = retrieve(
        query,
        top_k=1,
        per_product=True,
    )

    # 使用者有指定型號時，只保留該型號
    if has_variant:
        target_variant = get_target_variant(query)

        results = [
            result
            for result in results
            if target_variant
            in result["chunk"]["product"].upper()
        ]

    context = build_context(
        results,
        compare_variants=not has_variant,
    )

    forced_answer = None

    if not has_variant:
        forced_answer = build_forced_comparison_answer(
            results,
            query,
        )

    # ==================================================
    # 3. 如果 Python 已確認不同型號規格不同
    #    直接使用 deterministic structured response
    # ==================================================
    if forced_answer is not None:

        if get_response_language(query) == "zh":
            answer = "各型號規格如下：\n" + forced_answer
        else:
            answer = forced_answer

        print()
        print("=== Answer ===")
        print(answer)

        generated_tokens = llm.tokenize(
            answer.encode("utf-8"),
            add_bos=False,
        )

        token_count = len(generated_tokens)

        metrics = {
            "ttft": 0.0,
            "tps": 0.0,
            "generated_tokens": token_count,
            "generation_time": 0.0,
        }

        print()
        print("=== Performance ===")
        print("TTFT: N/A (deterministic structured response)")
        print(f"Generated Tokens: {token_count}")
        print("Generation Time: N/A")
        print("TPS: N/A")

        return answer, results, metrics

    # ==================================================
    # 4. 一般問題走 LLM generation
    # ==================================================

    response_language = get_response_language(query)

    if response_language == "en":
        language_instruction = """
IMPORTANT:
The user's question is in English.
Answer in English only.
Do not answer in Chinese.
"""
    else:
        language_instruction = """
重要：
使用者的問題包含中文。
請使用繁體中文回答。
"""

    system_prompt = """
你是一個 GIGABYTE AORUS MASTER 16 AM6H 產品規格助理。

只能根據提供的 Context 回答。

規則：

1. 不可以自行增加 Context 沒有的資訊。
2. 必須包含實際規格值。
3. 不同型號的資料不可混用。
4. 如果規格包含主要數值與備註，先回答主要數值。
5. 如果 Context 明確出現某功能，就不能回答不支援。
6. 回答簡潔。
"""

    user_prompt = f"""
{language_instruction}

Context:
{context}

Question:
{query}

請根據 Context 回答。
依照指定語言回答。
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

                print(
                    text,
                    end="",
                    flush=True,
                )

                answer_parts.append(text)

    print()

    answer = "".join(answer_parts)
    end_time = time.perf_counter()

    generated_tokens = llm.tokenize(
        answer.encode("utf-8"),
        add_bos=False,
    )

    token_count = len(generated_tokens)

    if first_token_time is not None:

        ttft = (
            first_token_time
            - start_time
        )

        generation_time = (
            end_time
            - first_token_time
        )

        if generation_time > 0:
            tps = (
                token_count
                / generation_time
            )
        else:
            tps = 0

    else:
        ttft = 0
        generation_time = 0
        tps = 0

    print()
    print("=== Performance ===")

    print(
        f"TTFT: "
        f"{ttft:.3f} seconds"
    )

    print(
        f"Generated Tokens: "
        f"{token_count}"
    )

    print(
        f"Generation Time: "
        f"{generation_time:.3f} seconds"
    )

    print(
        f"TPS: "
        f"{tps:.2f} tokens/second"
    )

    metrics = {
        "ttft": ttft,
        "tps": tps,
        "generated_tokens": token_count,
        "generation_time": generation_time,
    }

    return answer, results, metrics

if __name__ == "__main__":

    query = input(
        "請輸入問題："
    )

    answer, results, metrics = (
        answer_question(query)
    )

    print()
    print(
        "=== Retrieval Results ==="
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['chunk']['product']} | "
            f"{result['chunk']['category']} "
            f"(score="
            f"{result['final_score']:.4f})"
        )