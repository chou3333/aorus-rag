import json

from aorus_rag.retriever import retrieve


QUESTIONS_PATH = "questions.json"


with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)


top1_correct = 0
top3_correct = 0


print("開始 Retrieval Benchmark")
print("=" * 70)


for index, item in enumerate(questions, start=1):
    question = item["question"]
    expected_category = item["expected_category"]
    expected_product = item.get("expected_product")
    language = item["language"]

    results = retrieve(question, top_k=3)

    retrieved_items = [
        {
            "product": result["chunk"]["product"],
            "category": result["chunk"]["category"],
        }
        for result in results
    ]

    def is_correct(item_result):
        category_match = (
            item_result["category"] == expected_category
        )

        if expected_product is None:
            return category_match

        product_match = (
            item_result["product"] == expected_product
        )

        return category_match and product_match


    top1_hit = is_correct(retrieved_items[0])
    top3_hit = any(
        is_correct(result)
        for result in retrieved_items
    )

    if top1_hit:
        top1_correct += 1

    if top3_hit:
        top3_correct += 1

    print()
    print(f"Question {index}: {question}")
    print(f"Language: {language}")
    print(f"Expected Category: {expected_category}")

    if expected_product is not None:
        print(f"Expected Product: {expected_product}")

    print("Retrieved:")

    for rank, result in enumerate(retrieved_items, start=1):
        print(
            f"  Rank {rank}: "
            f"{result['product']} | "
            f"{result['category']}"
        )

    print(f"Top-1: {'PASS' if top1_hit else 'FAIL'}")
    print(f"Top-3: {'PASS' if top3_hit else 'FAIL'}")


total = len(questions)

top1_accuracy = top1_correct / total
top3_accuracy = top3_correct / total


print()
print("=" * 70)
print("Benchmark Result")
print("=" * 70)

print(f"Total Questions: {total}")

print(
    f"Top-1 Accuracy: "
    f"{top1_correct}/{total} "
    f"({top1_accuracy * 100:.2f}%)"
)

print(
    f"Top-3 Accuracy: "
    f"{top3_correct}/{total} "
    f"({top3_accuracy * 100:.2f}%)"
)