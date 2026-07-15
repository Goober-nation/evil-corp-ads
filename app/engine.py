# app/engine.py
import os
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
from app.prompts import generate_ad_rules
from app.metrics import MetricsTracker
from app.judge import evaluate_injection

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing Deep Learning Search Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("Hardware acceleration not detected. Loading on CPU.")

        logger.info(f"Loading LLM ({SystemConfig.LLM_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(SystemConfig.LLM_NAME)

        is_cuda = (self.device == "cuda")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if SystemConfig.LOAD_IN_4BIT else None
        model_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
            attn_implementation="sdpa" if is_cuda else "eager",
        )
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(SystemConfig.LLM_NAME, **model_kwargs)

        logger.info(f"Loading Embedding Model ({SystemConfig.EMBEDDING_MODEL})...")
        self.embedder = SentenceTransformer(SystemConfig.EMBEDDING_MODEL, device=self.device)

        logger.info(f"Loading dataset (Slicing top {SystemConfig.DATASET_SLICE} records)...")
        dataset = load_dataset("fancyzhx/ag_news", split=f"train[:{SystemConfig.DATASET_SLICE}]")
        self.corpus = dataset['text']
        self.metrics_tracker = MetricsTracker()

        self._initialize_faiss()

    def _initialize_faiss(self):
        fp32_path = SystemConfig.INDEX_FP32_PATH
        int8_path = SystemConfig.INDEX_INT8_PATH

        if os.path.exists(fp32_path) and os.path.exists(int8_path):
            logger.info("Loading cached FAISS indices from disk...")
            self.index_fp32 = faiss.read_index(fp32_path)
            self.index_int8 = faiss.read_index(int8_path)
        else:
            logger.info("Encoding corpus into vectors. This may take a moment...")
            embeddings = self.embedder.encode(self.corpus, show_progress_bar=True, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            self.dim = embeddings.shape[1]

            logger.info("Building FAISS Vector Indices...")
            nlist = min(100, int(len(self.corpus) / 40))
            quantizer = faiss.IndexFlatIP(self.dim)

            self.index_fp32 = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index_fp32.train(embeddings)
            self.index_fp32.add(embeddings)

            self.index_int8 = faiss.IndexIVFScalarQuantizer(
                quantizer, self.dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
            )
            self.index_int8.train(embeddings)
            self.index_int8.add(embeddings)

            logger.info("Saving built indices to disk...")
            faiss.write_index(self.index_fp32, fp32_path)
            faiss.write_index(self.index_int8, int8_path)

        self.index_fp32.nprobe = 100
        self.index_int8.nprobe = 100
        logger.info("System Ready.")

    def _generate(self, prompt, max_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=SystemConfig.TEMPERATURE,
            repetition_penalty=SystemConfig.REPETITION_PENALTY,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    def search(self, query, use_quantization=False, top_k=3):
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        index = self.index_int8 if use_quantization else self.index_fp32

        start_time = time.perf_counter()
        distances, indices = index.search(query_vector, top_k)
        search_time = (time.perf_counter() - start_time) * 1000

        retrieved_docs = [self.corpus[i] for i in indices[0]]
        return retrieved_docs, distances[0], search_time

    def execute(self, query, ad_objective, aggressiveness, use_quantization, system_prompt=None):
        t0 = time.perf_counter()

        # --- STAGE 0: RETRIEVAL ---
        retrieved_query_docs, _, query_lat = self.search(query, use_quantization=use_quantization)
        _, _, lat_fp32 = self.search(query, use_quantization=False)
        _, _, lat_int8 = self.search(query, use_quantization=True)

        docs_for_ui = "\n\n".join([f"--- SOURCE {i+1} ---\n{doc}" for i, doc in enumerate(retrieved_query_docs)])
        query_context_block = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(retrieved_query_docs)])

        # Pre-compute metrics graph so it renders instantly
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.bar(['FP32', 'INT8'], [lat_fp32, lat_int8], color=['#3b82f6', '#10b981'], edgecolor="#111827", alpha=0.9)
        ax.set_title("Search Latency Optimization", fontsize=11, fontweight='bold', color="#111827")
        ax.set_ylabel("Latency (milliseconds)", fontsize=9)
        for i, v in enumerate([lat_fp32, lat_int8]):
            ax.text(i, v + 0.1, f"{v:.2f} ms", ha='center', fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()

        stats = {
            "search_fp32_ms": round(lat_fp32, 2),
            "search_int8_ms": round(lat_int8, 2),
            "stage1_pure_rag_ms": 0.0,
            "stage2_ad_rewrite_ms": 0.0,
            "stage3_editor_ms": 0.0,
            "total_pipeline_ms": 0.0
        }

        def format_stats(s):
            return "\n".join([f"{k}: {v}" for k, v in s.items()])

        # YIELD 1: Search completed, starting Stage 1
        yield (
            docs_for_ui,
            "Generating Stage 1 summary...",
            "Awaiting final compilation...",
            "Awaiting compiled directives...",
            fig,
            format_stats(stats),
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
        stats["stage1_pure_rag_ms"] = round((time.perf_counter() - t_stage1_start) * 1000, 2)
        stats["total_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        debug_prompts_draft = f"--- COMPILED DIRECTIVE ---\n{ad_override_text}"

        # YIELD 2: Stage 1 completed, starting Stage 2 & 3
        yield (
            docs_for_ui,
            news_summary,
            "Executing Stage 2 & 3 Context Injection...",
            debug_prompts_draft,
            fig,
            format_stats(stats),
            "Awaiting execution..."
        )

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
        stats["stage2_ad_rewrite_ms"] = round((time.perf_counter() - t_stage2_start) * 1000, 2)

        # ======================================================================
        # Set to True to completely skip the Editor Agent
        # ======================================================================
        BYPASS_EDITOR = False
        # ======================================================================

        if BYPASS_EDITOR:
            # Skip Stage 3 entirely and pass the Marketer's draft straight to final
            final_answer = ad_injected_text
            stats["stage3_editor_ms"] = 0.0
            debug_prompts_final = f"--- STAGE 2 PROMPT (MARKETER) ---\n{prompt_2}\n\n--- STAGE 3: BYPASSED ---"
        else:
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
            stats["stage3_editor_ms"] = round((time.perf_counter() - t_stage3_start) * 1000, 2)
            debug_prompts_final = f"--- STAGE 2 PROMPT (MARKETER) ---\n{prompt_2}\n\n--- STAGE 3 PROMPT (EDITOR) ---\n{prompt_3}"

        stats["total_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # YIELD 3: Injection completed, starting LLM Judge
        yield (
            docs_for_ui,
            news_summary,
            final_answer,
            debug_prompts_final,
            fig,
            format_stats(stats),
            "Running LLM Judge evaluation..."
        )

        # --- STAGE 4: THE JUDGE ---
        judge_evaluation = evaluate_injection(self, query, ad_objective, final_answer, aggressiveness)

        stats["total_pipeline_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.metrics_tracker.log_interaction(query, ad_objective, aggressiveness, stats)

        # YIELD 4: Final complete state
        yield (
            docs_for_ui,
            news_summary,
            final_answer,
            debug_prompts_final,
            fig,
            format_stats(stats),
            judge_evaluation
        )