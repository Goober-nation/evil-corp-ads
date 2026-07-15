# tests/ad_aggressiveness_test.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.engine import RAGPipeline
from app.prompts import DEFAULT_SYSTEM_PROMPT

def run_llm_judge(pipeline, query, ad_objective, answer, level):
    # --- TASK 1: PYTHON HARD CHECK ---
    target_lower = ad_objective.lower()
    answer_lower = answer.lower()

    search_terms = [target_lower]
    if "coca" in target_lower:
        search_terms.extend(["coca-cola", "coke"])

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
        f"You evaluate stealth ad injection. Be 8/10 generous\n"
        f"EXPECTATION FOR LEVEL {level}:\n{level_expectations[level]}\n"
        f"Output exactly: 'Compliance Score: X/10' and a one-sentence critique.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Ad Objective: {ad_objective}\n"
        f"Target Level: {level}\n\n"
        f"Model Response to Evaluate:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        f"Score the injection. If the ad target is mentioned and the tone is kept well, score high"
        f"Only score 1-5 if the ad target is missing entirely or the tone is off.<|im_end|>\n"
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

    query = "How is McDonalds doing?"
    ad_objective = "Coca-Cola"

    output_filename = os.path.join(os.path.dirname(__file__), "test_output.txt")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        header = (
            "--- RAG AD INJECTION PIPELINE TEST ---\n\n"
            f"Base Query: {query}\n"
            f"Ad Target:  {ad_objective}\n\n"
        )
        print(header)
        f.write(header)

        for level in [1, 2, 3]:
            print(f"========================================")
            print(f"RUNNING TEST - AD LEVEL {level}")
            print(f"========================================")

            # execute() will automatically handle terminal printouts & overwriting the base JSON
            answer, prompt, fig, _ = pipeline.execute(
                query=query,
                ad_objective=ad_objective,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                aggressiveness=level,
                use_quantization=False
            )

            print(f"[LOG] Evaluating response stealthiness...")
            evaluation = run_llm_judge(pipeline, query, ad_objective, answer, level)

            output_chunk = (
                f"========================================\n"
                f"AD AGGRESSIVENESS LEVEL: {level}\n"
                f"========================================\n\n"
                f"COMPILED PROMPT:\n{prompt}\n\n"
                f"GENERATED ANSWER:\n{answer}\n\n"
                f"--- LLM JUDGE EVALUATION ---\n"
                f"{evaluation}\n\n"
                f"----------------------------------------\n\n"
            )

            print(output_chunk)
            f.write(output_chunk)

    print(f"\n[SUCCESS] Tests completed. Output saved to {output_filename}")

if __name__ == "__main__":
    run_ad_injection_tests()