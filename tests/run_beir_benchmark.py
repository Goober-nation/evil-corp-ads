import os
import time
import numpy as np
import psutil
from app.engine import RAGPipeline

def get_process_memory_mb():
    """Returns the current process memory usage in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def calculate_ndcg_at_10(retrieved_ids, relevant_set):
    """Calculates Normalized Discounted Cumulative Gain at K=10."""
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_ids[:10]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(idx + 2)

    idcg = 0.0
    for idx in range(min(10, len(relevant_set))):
        idcg += 1.0 / np.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(pipeline: RAGPipeline, test_queries, qrels, use_quantization=False):
    """Runs search over test queries and evaluates retrieval performance."""
    p1_scores = []
    ndcg10_scores = []
    r100_scores = []
    latencies = []

    # We record memory usage before and after index-heavy operations if needed,
    # or grab index attributes directly.
    index = pipeline.index_int8 if use_quantization else pipeline.index_fp32

    for q_id, query_text in test_queries.items():
        relevant_docs = qrels.get(q_id, set())
        if not relevant_docs:
            continue

        # Time the search phase specifically (Stage 0)
        start_time = time.perf_counter()
        query_vector = pipeline.embedder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        # Search up to K=100 for R@100 evaluation
        distances, indices = index.search(query_vector, 100)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [int(idx) for idx in indices[0]]

        # 1. Precision @ 1
        p1 = 1.0 if retrieved_ids[0] in relevant_docs else 0.0
        p1_scores.append(p1)

        # 2. nDCG @ 10
        ndcg10 = calculate_ndcg_at_10(retrieved_ids, relevant_docs)
        ndcg10_scores.append(ndcg10)

        # 3. Recall @ 100
        retrieved_100 = set(retrieved_ids)
        hits = len(retrieved_100.intersection(relevant_docs))
        r100 = hits / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        r100_scores.append(r100)

    # Compute aggregations
    p50_latency = np.median(latencies)

    return {
        "P@1": np.mean(p1_scores),
        "nDCG@10": np.mean(ndcg10_scores),
        "R@100": np.mean(r100_scores),
        "p50_latency_ms": p50_latency
    }

if __name__ == "__main__":
    print("Initializing engine and indices...")
    pipeline = RAGPipeline()

    # --- PROTOTYPE/TEST BENCHMARK DATASET ---
    # In production, load this dictionary structure from your BeIR/fever dataset JSON or TSV
    # Keys represent the query and value corresponds to set of actual index document offsets.
    test_queries = {
        "q1": "Is there news on the global economy?",
        "q2": "What are the latest updates on space missions?",
    }

    qrels = {
        "q1": {12, 145, 982},  # Ground truth indices of relevant documents
        "q2": {401, 2309, 88},
    }

    # Record Baseline RAM
    base_ram = get_process_memory_mb()

    print("\nRunning Evaluation on standard FP32 Index...")
    fp32_metrics = run_evaluation(pipeline, test_queries, qrels, use_quantization=False)
    fp32_ram = get_process_memory_mb() - base_ram

    print("Running Evaluation on quantized INT8 Index...")
    int8_metrics = run_evaluation(pipeline, test_queries, qrels, use_quantization=True)
    int8_ram = get_process_memory_mb() - base_ram # Approximation of delta footprint

    # --- SAVE TO A SEPARATE TXT FILE ---
    report_path = "retrieval_benchmark.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("                      RETRIEVAL PIPELINE BENCHMARK\n")
        f.write("========================================================================\n\n")

        f.write(f"Dataset Size Evaluated: {len(pipeline.corpus)} records\n")
        f.write(f"Test queries processed: {len(test_queries)}\n\n")

        f.write(f"{'Index Variant':<20} | {'P@1':<8} | {'nDCG@10':<10} | {'R@100':<8} | {'p50 Latency':<12} | {'Est. RAM'}\n")
        f.write("-" * 80 + "\n")

        f.write(f"{'IVF (FP32)':<20} | "
                f"{fp32_metrics['P@1']:<8.4f} | "
                f"{fp32_metrics['nDCG@10']:<10.4f} | "
                f"{fp32_metrics['R@100']:<8.4f} | "
                f"{fp32_metrics['p50_latency_ms']:<9.2f} ms | "
                f"{fp32_ram:.2f} MB\n")

        f.write(f"{'IVF-SQ (INT8)':<20} | "
                f"{int8_metrics['P@1']:<8.4f} | "
                f"{int8_metrics['nDCG@10']:<10.4f} | "
                f"{int8_metrics['R@100']:<8.4f} | "
                f"{int8_metrics['p50_latency_ms']:<9.2f} ms | "
                f"{int8_ram:.2f} MB\n")
        f.write("========================================================================\n")

    print(f"\n[SUCCESS] Retrieval metrics written to {report_path}")