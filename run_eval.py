"""
Run the RAG evaluation against the Tesla 10-K golden set.

Usage:
    1. Make sure your Tesla 10-K PDF path is correct below.
    2. Run:  python run_eval.py

It loads the document once, runs every question through answer_question(),
and uses your Groq model as an LLM judge to grade each answer semantically
(so "12,556" vs "$12.5 billion" both count as correct).
"""
import time
from main import build_vectorstore, answer_question, model
from eval_set import eval_set

# Path to the Tesla 10-K you've been testing with
PDF_PATH = r"C:\Users\Pranav\Downloads\nasdaq-tsla-2023-10K-23570030.pdf"   # <-- change to your actual file path


def judge(question, expected, actual):
    """Use the LLM to decide if `actual` conveys the same info as `expected`."""
    prompt = f"""You are grading a RAG system's answer against a reference answer.

Question: {question}
Reference answer: {expected}
System answer: {actual}

Does the system answer convey the same key information as the reference answer?
Ignore differences in wording, formatting, or number style (e.g. "12,556 million"
and "$12.5 billion" are the same). Reply with ONLY one word: CORRECT or INCORRECT."""

    response = model.invoke([{"role": "user", "content": prompt}])
    return "CORRECT" in response.content.upper()


def main():
    print("Loading document and building retrievers...")
    build_vectorstore(PDF_PATH)

    correct = 0
    failures = []

    for i, item in enumerate(eval_set, 1):
        answer = answer_question(item["question"])
        passed = judge(item["question"], item["expected"], answer)

        time.sleep(10)

        if passed:
            correct += 1
            print(f"[{i:2}] PASS  {item['question']}")
        else:
            print(f"[{i:2}] FAIL  {item['question']}")
            failures.append((item["question"], item["expected"], answer))

    total = len(eval_set)
    accuracy = correct / total * 100
    print("\n" + "=" * 50)
    print(f"Accuracy: {accuracy:.1f}%  ({correct}/{total})")
    print("=" * 50)

    if failures:
        print("\nFailed questions (for review):")
        for q, expected, actual in failures:
            print(f"\nQ: {q}")
            print(f"   expected: {expected}")
            print(f"   got:      {actual[:160]}...")


if __name__ == "__main__":
    main()
