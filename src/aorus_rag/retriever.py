import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "embeddings/embeddings.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


CATEGORY_ALIASES = {
    "作業系統": [
        "作業系統",
        "os",
        "operating system",
        "windows",
    ],
    "中央處理器": [
        "中央處理器",
        "處理器",
        "cpu",
        "processor",
    ],
    "顯示晶片": [
        "顯示晶片",
        "顯示卡",
        "gpu",
        "graphics",
        "graphics card",
    ],
    "顯示器": [
        "顯示器",
        "螢幕",
        "screen",
        "display",
        "monitor",
    ],
    "記憶體": [
        "記憶體",
        "ram",
        "memory",
    ],
    "儲存裝置": [
        "儲存",
        "儲存裝置",
        "硬碟",
        "ssd",
        "storage",
    ],
    "鍵盤種類": [
        "鍵盤",
        "keyboard",
    ],
    "連接埠": [
        "連接埠",
        "接口",
        "port",
        "ports",
        "usb",
        "thunderbolt",
        "hdmi",
    ],
    "音效": [
        "音效",
        "喇叭",
        "speaker",
        "audio",
        "sound",
    ],
    "通訊": [
        "通訊",
        "wifi",
        "wi-fi",
        "bluetooth",
        "網路",
        "network",
        "lan",
    ],
    "視訊鏡頭": [
        "視訊鏡頭",
        "攝影機",
        "鏡頭",
        "webcam",
        "camera",
    ],
    "安全裝置": [
        "安全裝置",
        "安全",
        "tpm",
        "security",
    ],
    "電池": [
        "電池",
        "battery",
    ],
    "變壓器": [
        "變壓器",
        "充電器",
        "adapter",
        "charger",
    ],
    "尺寸": [
        "尺寸",
        "dimension",
        "dimensions",
        "size",
    ],
    "重量": [
        "重量",
        "weight",
        "kg",
    ],
    "顏色": [
        "顏色",
        "color",
        "colour",
    ],
}


with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)


embeddings = np.load(EMBEDDINGS_PATH)
if len(embeddings) != len(chunks):
    raise ValueError("Chunks 與 embeddings 數量不符，請重新執行 python -m aorus_rag.embedder")


print("正在載入 Embedding 模型...")
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)


def calculate_alias_bonus(query, category):
    query_lower = query.lower()

    aliases = CATEGORY_ALIASES.get(category, [])

    for alias in aliases:
        pattern = re.escape(alias.lower())
        if alias.isascii():
            pattern = rf"(?<![a-z0-9]){pattern}(?![a-z0-9])"
        if re.search(pattern, query_lower):
            return 0.35

    return 0.0


def retrieve(query, top_k=3, *, per_product=False):

    products = list(dict.fromkeys(chunk["product"] for chunk in chunks))
    requested = [product for product in products if re.search(
        rf"(?<![a-z0-9]){re.escape(product.split()[-1])}(?![a-z0-9])",
        query, re.IGNORECASE,
    )]
    # Model codes are metadata filters, not useful semantic specification terms.
    semantic_query = query
    for product in requested:
        semantic_query = re.sub(re.escape(product), "", semantic_query, flags=re.IGNORECASE)
        semantic_query = re.sub(
            rf"(?<![a-z0-9]){re.escape(product.split()[-1])}(?![a-z0-9])",
            "", semantic_query, flags=re.IGNORECASE,
        )
    query_embedding = model.encode(
        semantic_query.strip() or query,
        normalize_embeddings=True
    )

    semantic_scores = embeddings @ query_embedding

    final_scores = []

    for index, chunk in enumerate(chunks):

        semantic_score = float(semantic_scores[index])

        category = chunk["category"]

        alias_bonus = calculate_alias_bonus(
            query,
            category
        )

        final_score = semantic_score + alias_bonus

        final_scores.append(final_score)

    final_scores = np.array(final_scores)

    # Exact model filtering prevents similar model names from crossing over.
    selected = requested or products
    top_indices = []
    for product in selected:
        ranked = [int(i) for i in np.argsort(final_scores)[::-1]
                  if chunks[i]["product"] == product]
        top_indices.extend(ranked[:max(0, top_k)])
    if not per_product:
        top_indices = sorted(top_indices, key=lambda i: final_scores[i], reverse=True)[:max(0, top_k)]

    results = []

    for index in top_indices:

        result = {
            "semantic_score": float(semantic_scores[index]),
            "final_score": float(final_scores[index]),
            "chunk": chunks[index]
        }

        results.append(result)

    return results


if __name__ == "__main__":

    query = input("請輸入問題：")

    results = retrieve(query)

    print()
    print("搜尋結果：")

    for rank, result in enumerate(results, start=1):

        print()
        print("=" * 60)
        print(f"Rank {rank}")
        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )
        print(
            f"Final Score: "
            f"{result['final_score']:.4f}"
        )
        print(result["chunk"]["text"])
