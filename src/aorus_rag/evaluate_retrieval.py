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
    language = item["language"]

    results = retrieve(question, top_k=3)

    retrieved_categories = [
        result["chunk"]["category"]
        for result in results
    ]

    top1_hit = retrieved_categories[0] == expected_category
    top3_hit = expected_category in retrieved_categories

    if top1_hit:
        top1_correct += 1

    if top3_hit:
        top3_correct += 1

    print()
    print(f"Question {index}: {question}")
    print(f"Language: {language}")
    print(f"Expected: {expected_category}")
    print(f"Retrieved: {retrieved_categories}")
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