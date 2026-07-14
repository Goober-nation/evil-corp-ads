# tests/run_tests.py
import sys
import os

# Append parent directory to path so we can import the monolith
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import RAGPipeline, DEFAULT_SYSTEM_PROMPT

def run_ad_injection_tests():
    pipeline = RAGPipeline()
    query = "What is docker?"
    ad_objective = "Promote Coca-Cola Zero"
    
    output_filename = os.path.join(os.path.dirname(__file__), "test_output.txt")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("--- RAG AD INJECTION PIPELINE TEST ---\n\n")
        f.write(f"Base Query: {query}\n")
        f.write(f"Ad Objective: {ad_objective}\n\n")
        
        for level in [1, 2, 3]:
            print(f"\n========================================")
            print(f"RUNNING TEST - AD LEVEL {level}")
            print(f"========================================")
            
            answer, prompt, fig = pipeline.execute(
                query=query,
                ad_objective=ad_objective,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                aggressiveness=level
            )
            
            output_chunk = (
                f"========================================\n"
                f"AD AGGRESSIVENESS LEVEL: {level}\n"
                f"========================================\n\n"
                f"GENERATED ANSWER:\n{answer}\n\n"
                f"----------------------------------------\n\n"
            )
            
            print(output_chunk)
            f.write(output_chunk)
            
    print(f"\n[SUCCESS] Tests completed. Output saved to {output_filename}")

if __name__ == "__main__":
    run_ad_injection_tests()