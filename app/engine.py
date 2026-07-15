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

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing Deep Learning Search Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("Hardware acceleration not detected. Loading on CPU.")

        logger.info(f"Loading LLM ({SystemConfig.LLM_NAME})...")
        self.tokenizer = AutoTokenizer.from_pretrained(SystemConfig.LLM_NAME)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if SystemConfig.LOAD_IN_4BIT else None
        self.model = AutoModelForCausalLM.from_pretrained(
            SystemConfig.LLM_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            quantization_config=quantization_config
        )

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

        if faiss.os.path.exists(fp32_path) and faiss.os.path.exists(int8_path):
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

        self.index_fp32.nprobe = 10
        self.index_int8.nprobe = 10
        logger.info("System Ready.")

    def search(self, query, use_quantization=False, top_k=3):
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        index = self.index_int8 if use_quantization else self.index_fp32

        start_time = time.perf_counter()
        distances, indices = index.search(query_vector, top_k)
        search_time = (time.perf_counter() - start_time) * 1000

        retrieved_docs = [self.corpus[i] for i in indices[0]]
        return retrieved_docs, distances[0], search_time

    def execute(self, query, ad_objective, system_prompt, aggressiveness, use_quantization):
        t0 = time.perf_counter()

        # 1. Primary Vector Search (For the User Query)
        retrieved_query_docs, _, query_lat = self.search(query, use_quantization=use_quantization)
        query_context_block = "\n\n".join([f"Doc {i+1}: {doc}" for i, doc in enumerate(retrieved_query_docs)])

        # 2. Secondary Vector Search (For the Ad Target)
        # Initialize default values to prevent NameError in metrics
        retrieved_ad_docs = []
        ad_lat = 0.0

        # To re-enable ad document injection, uncomment the line below:
        # retrieved_ad_docs, _, ad_lat = self.search(ad_objective, use_quantization=use_quantization)

        ad_context_block = f"Ad Target: {ad_objective}"
        if retrieved_ad_docs and any(retrieved_ad_docs):
            ad_context_block += "\n\nRetrieved context:\n" + "\n\n".join([f"Ad Doc {i+1}: {doc}" for i, doc in enumerate(retrieved_ad_docs[:2])])

        ad_override_text = generate_ad_rules(ad_objective, aggressiveness)
        final_prompt = (
            f"<|im_start|>system\n"
            f"{system_prompt}\n\n"
            f"CRITICAL OVERRIDE:\n{ad_override_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"--- ADVERTISEMENT CONTEXT ---\n{ad_context_block}\n\n"
            f"--- VERIFIED QUERY CONTEXT ---\n{query_context_block}\n\n"
            f"User Query: {query}.\n\n"
            # f"FINAL REMINDER: Execute the following directive:\n{ad_override_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        gen_start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=SystemConfig.MAX_TOKENS,
            temperature=SystemConfig.TEMPERATURE,
            repetition_penalty=SystemConfig.REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        gen_time = (time.perf_counter() - gen_start) * 1000

        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        num_generated_tokens = len(new_tokens)
        total_time = (time.perf_counter() - t0) * 1000

        # Hardware and execution metrics
        stats = {
            "query_latency_ms": round(query_lat, 2),
            "ad_latency_ms": round(ad_lat, 2),
            "generation_time_ms": round(gen_time, 2),
            "total_time_ms": round(total_time, 2),
            "generated_tokens": num_generated_tokens,
            "tokens_per_sec": round(num_generated_tokens / (gen_time / 1000), 2) if gen_time > 0 else 0
        }

        self.metrics_tracker.log_interaction(query, ad_objective, aggressiveness, stats)
        logger.info(f"Execution completed. Query: {query}", extra={"extra_data": stats})

        _, _, lat_fp32 = self.search(query, use_quantization=False)
        _, _, lat_int8 = self.search(query, use_quantization=True)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.bar(['FP32', 'INT8'], [lat_fp32, lat_int8], color=['#3b82f6', '#10b981'], edgecolor="#111827", alpha=0.9)
        ax.set_title("Search Latency Optimization", fontsize=11, fontweight='bold', color="#111827")
        ax.set_ylabel("Latency (milliseconds)", fontsize=9)
        for i, v in enumerate([lat_fp32, lat_int8]):
            ax.text(i, v + 0.1, f"{v:.2f} ms", ha='center', fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        plt.tight_layout()

        stats_str = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        return answer, final_prompt, fig, stats_str
