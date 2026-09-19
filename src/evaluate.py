"""系统评估: 检索召回率 + (可选)生成答案导出供人工核对。

用法:
    python src/evaluate.py                # 检索召回率评估(Top-3)
    python src/evaluate.py --with-llm     # 同时生成答案, 保存到 data/eval_answers.json
                                          # 供人工核对数值与引用准确率
测试题集: data/test_questions.json
    expected_sources 为来源文件名关键词, 任一命中即视为该题检索成功。
"""

import argparse
import json
import time
from pathlib import Path

from qa import RAGEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "test_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="同时生成答案供人工核对")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    engine = RAGEngine(top_k=args.top_k)

    hits_count = 0
    details = []
    for q in questions:
        hits = engine.retrieve(q["question"], top_k=args.top_k)
        sources = [h["source"] for h in hits]
        ok = any(
            any(kw.lower() in s.lower() for s in sources)
            for kw in q["expected_sources"]
        )
        hits_count += ok
        details.append({
            "question": q["question"],
            "recall_hit": ok,
            "retrieved": sources,
            "expected": q["expected_sources"],
        })
        mark = "OK " if ok else "MISS"
        print(f"[{mark}] {q['question']}")
        if not ok:
            print(f"       期望: {q['expected_sources']}")
            print(f"       实得: {sources}")

    recall = hits_count / len(questions) * 100
    print(f"\nTop-{args.top_k} 检索召回率: {hits_count}/{len(questions)} = {recall:.0f}%"
          f" (目标 >= 80%)")

    report = {"recall_at_k": recall, "top_k": args.top_k, "details": details}

    if args.with_llm:
        answers = []
        for q in questions:
            result = engine.ask(q["question"])
            answers.append(result)
            print(f"已生成: {q['question']} ({result['elapsed_s']}s)")
            time.sleep(1)
        out = PROJECT_ROOT / "data" / "eval_answers.json"
        out.write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"答案已保存到 {out}, 请人工核对数值与引用是否和原文一致")
        report["answers_file"] = str(out)

    report_path = PROJECT_ROOT / "data" / "eval_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评估报告 -> {report_path}")


if __name__ == "__main__":
    main()
