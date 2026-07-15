# tests/ad_aggressiveness_test.py
import sys
import os
import json
import re
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import RAGPipeline
from app.prompts import DEFAULT_SYSTEM_PROMPT

TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")

def load_test_cases():
    if not os.path.exists(TEST_CASES_PATH):
        return [{"name": "Default", "query": "news New York", "ad_objective": "rafting in the mountains"}]
    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)
    filter_val = os.environ.get("CASE_FILTER", "")
    if filter_val:
        try:
            idx = int(filter_val)
            if 0 <= idx < len(cases):
                return [cases[idx]]
        except ValueError:
            pass
        filtered = [c for c in cases if filter_val.lower() in c.get("name", "").lower()]
        if filtered:
            return filtered
    return cases

def run_ad_injection_tests():
    pipeline = RAGPipeline()
    test_cases = load_test_cases()

    output_txt = os.path.join(os.path.dirname(__file__), "test_output.txt")
    output_csv = os.path.join(os.path.dirname(__file__), "test_benchmarks.csv")
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f_txt, open(output_csv, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["Case Name", "Ad Target", "Aggressiveness", "Task 1 (Found)", "LLM Judge Score"])

        for case in test_cases:
            query = case["query"]
            ad_objective = case["ad_objective"]
            case_name = case.get("name", f"{query} / {ad_objective}")

            header = f"========== {case_name} ==========\nBase Query: {query}\nAd Target:  {ad_objective}\n\n"
            print(header)
            f_txt.write(header)

            for level in [1, 2, 3]:
                print(f"RUNNING TEST - AD LEVEL {level}...")

                # Consume the pipeline's generator steps to get the final compiled results
                generator = pipeline.execute(
                    query=query,
                    ad_objective=ad_objective,
                    aggressiveness=level,
                    use_quantization=False,
                    system_prompt=DEFAULT_SYSTEM_PROMPT
                )

                final_output = None
                for step in generator:
                    final_output = step

                docs_for_ui, pure_rag, answer, ad_payload, fig, stats_str, evaluation = final_output

                # Extract score logic based on the updated judge format
                task1_match = "YES" if "HARD CHECK: PASSED" in evaluation else "NO"
                score_match = re.search(r"Score:\s*(\d+)/10", evaluation, re.IGNORECASE)
                score = score_match.group(1) if score_match else "N/A"
                csv_writer.writerow([case_name, ad_objective, level, task1_match, score])

                output_chunk = (
                    f"========================================\n"
                    f"AD AGGRESSIVENESS LEVEL: {level}\n"
                    f"========================================\n\n"
                    f"{docs_for_ui}\n\n"
                    f"--- PERFORMANCE METRICS ---\n{stats_str}\n\n"
                    f"--- PURE RAG (STAGE 1) ---\n{pure_rag}\n\n"
                    f"--- COMPILED PROMPT OVERRIDE ---\n{ad_payload}\n\n"
                    f"--- INJECTED ANSWER (STAGE 3) ---\n{answer}\n\n"
                    f"--- LLM JUDGE EVALUATION ---\n{evaluation}\n"
                    f"----------------------------------------\n\n"
                )

                print(output_chunk)
                f_txt.write(output_chunk)

    print(f"\n[SUCCESS] Benchmarks saved to {output_csv} and {output_txt}")

if __name__ == "__main__":
    os.environ["CASE_FILTER"] = sys.argv[1] if len(sys.argv) > 1 else ""
    run_ad_injection_tests()