import json


INPUT_PATH = "data/specs.json"
OUTPUT_PATH = "data/chunks.json"

PRODUCT_NAME = "AORUS MASTER 16 AM6H"


with open(INPUT_PATH, "r", encoding="utf-8") as file:
    specs = json.load(file)


chunks = []


for index, (category, values) in enumerate(specs.items()):

    content = "\n".join(values)

    chunk = {
        "id": index,
        "product": PRODUCT_NAME,
        "category": category,
        "content": content,
        "text": (
            f"產品：{PRODUCT_NAME}\n"
            f"規格類別：{category}\n"
            f"內容：{content}"
        )
    }

    chunks.append(chunk)


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


print("Chunking 完成")
print(f"總共建立 {len(chunks)} 個 chunks")
print(f"已輸出到：{OUTPUT_PATH}")


for chunk in chunks[:5]:
    print()
    print("=" * 50)
    print("Chunk ID:", chunk["id"])
    print(chunk["text"])