import json
import csv

from aorus_rag.rag import answer_question


QUESTIONS_PATH = "questions.json"
OUTPUT_PATH = "rag_benchmark_results.csv"


with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
    questions = json.load(file)


results_rows = []

total_ttft = 0
total_tps = 0
llm_metric_count = 0


print("開始 End-to-End RAG Benchmark")
print("=" * 70)


for index, item in enumerate(questions, start=1):
    question = item["question"]
    expected_category = item["expected_category"]
    language = item["language"]

    print()
    print(f"[{index}/{len(questions)}]")
    print("Question:", question)

    answer, retrieval_results, metrics = answer_question(question)

    retrieved_categories = [
        result["chunk"]["category"]
        for result in retrieval_results
    ]

    top1_category = retrieved_categories[0] if retrieved_categories else None

    retrieval_correct = (
        top1_category == expected_category
    )

    # 判斷是否真的有經過 LLM generation
    used_llm = (
        metrics.get("used_llm", metrics["generation_time"] > 0)
    )

    row = {
        "question": question,
        "language": language,
        "expected_category": expected_category,
        "top1_category": top1_category,
        "retrieval_correct": retrieval_correct,
        "answer": answer,
        "used_llm": used_llm,
        "ttft_seconds": (
            metrics["ttft"]
            if used_llm
            else ""
        ),
        "generated_tokens": metrics["generated_tokens"],
        "generation_time_seconds": (
            metrics["generation_time"]
            if used_llm
            else ""
        ),
        "tps": (
            metrics["tps"]
            if used_llm
            else ""
        ),
    }

    results_rows.append(row)

    # 只統計真正經過 LLM generation 的題目
    if used_llm:
        total_ttft += metrics["ttft"]
        total_tps += metrics["tps"]
        llm_metric_count += 1


if llm_metric_count > 0:
    average_ttft = total_ttft / llm_metric_count
    average_tps = total_tps / llm_metric_count
else:
    average_ttft = 0
    average_tps = 0


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "question",
        "language",
        "expected_category",
        "top1_category",
        "retrieval_correct",
        "answer",
        "used_llm",
        "ttft_seconds",
        "generated_tokens",
        "generation_time_seconds",
        "tps",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results_rows)


print()
print("=" * 70)
print("RAG Benchmark Result")
print("=" * 70)

print(f"Total Questions: {len(results_rows)}")
print(f"LLM-generated Questions: {llm_metric_count}")
print(
    f"Deterministic Questions: "
    f"{len(results_rows) - llm_metric_count}"
)
print(
    f"Average LLM TTFT: "
    f"{average_ttft:.3f} seconds"
)
print(
    f"Average LLM TPS: "
    f"{average_tps:.2f} tokens/second"
)
print(f"Results saved to: {OUTPUT_PATH}")