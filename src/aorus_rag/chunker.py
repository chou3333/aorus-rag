import json
from pathlib import Path

INPUT_PATH = "data/specs.json"
OUTPUT_PATH = "data/chunks.json"


def build_chunks(products):
    chunks = []
    for product, specs in products.items():
        for category, values in specs.items():
            content = "\n".join(values)
            chunks.append({
                "id": len(chunks),
                "product": product,
                "category": category,
                "content": content,
                "text": f"產品：{product}\n規格類別：{category}\n內容：{content}",
            })
    return chunks


def main():
    products = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    chunks = build_chunks(products)
    Path(OUTPUT_PATH).write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已建立 {len(chunks)} 個 chunks，輸出到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
