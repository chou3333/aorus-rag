import json
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_PATH = "data/chunks.json"
OUTPUT_PATH = "embeddings/embeddings.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)


texts = [chunk["text"] for chunk in chunks]


print("正在載入 Embedding 模型...")
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)


print("正在產生 embeddings...")
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


np.save(OUTPUT_PATH, embeddings)


print()
print("Embedding 完成")
print("Chunk 數量：", len(chunks))
print("Embedding shape：", embeddings.shape)
print("已輸出到：", OUTPUT_PATH)