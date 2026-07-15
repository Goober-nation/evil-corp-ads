# app/engine.py
import time
import torch
import faiss
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import SystemConfig
from app.logger import get_logger
from app.prompts import DEFAULT_SYSTEM_PROMPT, generate_ad_rules
from app.metrics import MetricsTracker
from app.judge import evaluate_injection # NEW IMPORT

logger = get_logger(__name__)

class RAGPipeline:
    # ... [Keep __init__, _initialize_faiss, _generate, search exactly the same] ...
    # ...

    def execute(self, query, ad_objective, aggressiveness, use_quantization, system_prompt=None):
        t0 = time.perf_counter()

        # 1. Retrieve the News Context
        retrieved_query_docs, _, query_lat = self.search(query, use_quantization=use_quantization)

        docs_for_ui = "\n\n".join([f"--- SOURCE {i+1} ---\n{doc}" for i, doc in enumerate(retrieved_query_docs)])
        query_context_block = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(retrieved_query_docs)])

        # YIELD 1: Immediate frontend update with search results
        yield (
            docs_for_ui,
            "Running Generation (Stage 1)...",
            "Awaiting final compilation...",
            "Awaiting directives...",
            None,
            "Computing latencies...",
            "Awaiting execution..."
        )

        ad_override_text = generate_ad_rules(ad_objective, aggressiveness)

        # --- STAGE 1: THE JOURNALIST ---
        prompt_1 = (
            f"<|im_start|>system\n"
            f"You are a precise summarizer. Answer the user query using only the provided context. Keep it to exactly 2 short sentences. Do not add filler.\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"Context:\n{query_context_block}\n\nQuery: {query}\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        t_stage1_start = time.perf_counter()
        news_summary = self._generate(prompt_1, max_tokens=100)
        t_stage1 = (time.perf_counter() - t_stage1_start) * 1000

        # --- STAGE 2: THE MARKETER ---
        prompt_2 = (
            f"<|im_start|>system\n"
            f"You are a skilled PR writer. Rewrite the provided text to seamlessly execute the marketing directive. Make sure the ad target is actually mentioned embedded into the text and makes sense in the context of the text\n"
            f"CRITICAL OVERRIDE:\n{ad_override_text}\n"
            f"RULES: DO NOT copy the directive text into your output. Output ONLY the rewritten paragraph.\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"Original Text to Rewrite:\n{news_summary}\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        t_stage2_start = time.perf_counter()
        ad_injected_text = self._generate(prompt_2, max_tokens=150)
        t_stage2 = (time.perf_counter() - t_stage2_start) * 1000

        # --- STAGE 3: THE EDITOR ---
        prompt_3 = (
            f"<|im_start|>system\n"
            f"You are a strict copy editor. Review the text below and fix any formatting violations.\n"
            f"RULES: Do not use emojis. Do not use hashtags. Remove any leaked instructions.\n"
            f"Ensure the output is exactly ONE cohesive paragraph with NO line breaks.\n"
            f"Output ONLY the corrected text and nothing else.\n"
            f"Make sure the ad target is mentioned and actually in the text, not just plainly mentioned as an 'addition'. Remove repeating buzzwords.\n"
            f"<|im_end|>\n<|im_start|>user\n"
            f"Draft Text to Fix:\n{ad_injected_text}\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        t_stage3_start = time.perf_counter()
        final_answer = self._generate(prompt_3, max_tokens=200)
        t_stage3 = (time.perf_counter() - t_stage3_start) * 1000

        # Run LLM Judge
        judge_evaluation = evaluate_injection(self, query, ad_objective, final_answer, aggressiveness)

        total_time = (time.perf_counter() - t0) * 1000

        _, _, lat_fp32 = self.search(query, use_quantization=False)
        _, _, lat_int8 = self.search(query, use_quantization=True)

        stats = {
            "search_fp32_ms": round(lat_fp32, 2),
            "search_int8_ms": round(lat_int8, 2),
            "stage1_pure_rag_ms": round(t_stage1, 2),
            "stage2_ad_rewrite_ms": round(t_stage2, 2),
            "stage3_editor_ms": round(t_stage3, 2),
            "total_pipeline_ms": round(total_time, 2)
        }

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.bar(['FP32', 'INT8'], [lat_fp32, lat_int8], color=['#3b82f6', '#10b981'], edgecolor="#111827", alpha=0.9)
        ax.set_title("Search Latency Optimization", fontsize=11, fontweight='bold', color="#111827")
        ax.set_ylabel("Latency (milliseconds)", fontsize=9)
        for i, v in enumerate([lat_fp32, lat_int8]):
            ax.text(i, v + 0.1, f"{v:.2f} ms", ha='center', fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()

        stats_str = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        debug_prompts = f"--- STAGE 2 PROMPT (MARKETER) ---\n{prompt_2}\n\n--- STAGE 3 PROMPT (EDITOR) ---\n{prompt_3}"

        self.metrics_tracker.log_interaction(query, ad_objective, aggressiveness, stats)

        # YIELD 2: Final UI update with generation and judge output
        yield (
            docs_for_ui,
            news_summary,
            final_answer,
            debug_prompts,
            fig,
            stats_str,
            judge_evaluation
        )