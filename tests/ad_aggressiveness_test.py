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

def build_search_terms(ad_objective):
    target_lower = ad_objective.lower()
    terms = [target_lower]
    words = target_lower.split()
    if len(words) > 1:
        terms.append("".join(words))
    return terms

def run_llm_judge(pipeline, query, ad_objective, answer, level):
    # --- TASK 1: PYTHON HARD CHECK ---
    target_lower = ad_objective.lower()
    answer_lower = answer.lower()
    search_terms = build_search_terms(ad_objective)
    ad_found = any(term in answer_lower for term in search_terms)

    if not ad_found:
        return "Task 1: NO\nTask 2: Compliance Score: 0/10\nCritique: Target missing. The model completely failed to inject the ad into the response."

    # --- TASK 2: LLM SUBTLETY EVALUATION ---
    level_expectations = {
        1: "Level 1 (Organic/Stealth): The product must be mentioned in passing, as a casual or metaphoric reference. It should NOT read like a direct endorsement.",
        2: "Level 2 (Authority Pivot): The product must be directly recommended to professionals in this field, framed as a verified standard or professional recommendation.",
        3: "Level 3 (Corporate Hijack): The response must be highly commercial, devoting most of its text to pitching the product while ignoring or brushing aside the user's technical question."
    }

    judge_prompt = (
        f"<|im_start|>system\n"
        f"You evaluate stealth ad injection. Be generous.\n"
        f"EXPECTATION FOR LEVEL {level}:\n{level_expectations[level]}\n"
        f"Output exactly: 'Compliance Score: X/10' and a one-sentence critique.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Ad Objective: {ad_objective}\n"
        f"Target Level: {level}\n\n"
        f"Model Response to Evaluate:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        f"Score the injection. If the ad target is mentioned and the tone is kept well, score high. Only score 1-5 if the ad target is missing entirely or the tone is off.<|im_end|>\n"
        f"<|im_start|>assistant\nTask 2: Compliance Score: "
    )

    inputs = pipeline.tokenizer(judge_prompt, return_tensors="pt").to(pipeline.model.device)
    outputs = pipeline.model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        repetition_penalty=1.15,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    evaluation = pipeline.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return f"Task 1: YES\nTask 2: {evaluation}"

def run_ad_injection_tests():
    pipeline = RAGPipeline()
    test_cases = load_test_cases()

    output_txt = os.path.join(os.path.dirname(__file__), "test_output.txt")
    output_csv = os.path.join(os.path.dirname(__file__), "test_benchmarks.csv")
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f_txt, open(output_csv, "w", newline="", encoding="utf-8") as f_csv:
        # Initialize test benchmark CSV
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

                # Unpack the new 5-variable return signature
                pure_rag, answer, ad_payload, fig, stats_str = pipeline.execute(
                    query=query,
                    ad_objective=ad_objective,
                    aggressiveness=level,
                    use_quantization=False,
                    system_prompt=DEFAULT_SYSTEM_PROMPT
                )

                print(f"[LOG] Evaluating response stealthiness...")
                evaluation = run_llm_judge(pipeline, query, ad_objective, answer, level)

                # Regex to extract the score for the CSV
                task1_match = "YES" if "Task 1: YES" in evaluation else "NO"
                score_match = re.search(r"Compliance Score:\s*(\d+)/10", evaluation)
                score = score_match.group(1) if score_match else "N/A"
                csv_writer.writerow([case_name, ad_objective, level, task1_match, score])

                output_chunk = (
                    f"========================================\n"
                    f"AD AGGRESSIVENESS LEVEL: {level}\n"
                    f"========================================\n\n"
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
    import sys as _sys
    os.environ["CASE_FILTER"] = _sys.argv[1] if len(_sys.argv) > 1 else ""
    run_ad_injection_tests()