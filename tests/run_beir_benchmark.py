import os
import sys
import time
import faiss
import numpy as np
import psutil
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import RAGPipeline
from app.config import SystemConfig

def get_process_memory_mb():
    """Returns the current process memory usage in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def calculate_ndcg_at_10(retrieved_ids, relevance_map):
    """Calculates nDCG@10 with graded relevance."""
    ideal = sorted(relevance_map.values(), reverse=True)[:10]

    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_ids[:10]):
        rel = relevance_map.get(doc_id, 0.0)
        dcg += rel / np.log2(idx + 2)

    idcg = 0.0
    for idx, rel in enumerate(ideal):
        idcg += rel / np.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(pipeline: RAGPipeline, test_queries, qrels, qrel_grades, use_quantization=False):
    """Runs search over test queries and evaluates retrieval performance."""
    p10_scores = []
    ndcg10_scores = []
    r10_scores = []
    mrr_scores = []
    latencies = []

    index = pipeline.index_int8 if use_quantization else pipeline.index_fp32

    for q_id, query_text in test_queries.items():
        relevant_docs = qrels.get(q_id, set())
        if not relevant_docs:
            continue

        start_time = time.perf_counter()
        query_vector = pipeline.embedder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, 10)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [int(idx) for idx in indices[0]]

        # 1. Precision @ 10
        hits_10 = sum(1 for doc_id in retrieved_ids[:10] if doc_id in relevant_docs)
        p10_scores.append(hits_10 / 10)

        # 2. nDCG @ 10 (graded relevance)
        ndcg10 = calculate_ndcg_at_10(retrieved_ids, qrel_grades[q_id])
        ndcg10_scores.append(ndcg10)

        # 3. Recall @ 10
        retrieved_set = set(retrieved_ids)
        hits = len(retrieved_set.intersection(relevant_docs))
        r10 = hits / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        r10_scores.append(r10)

        # 4. MRR
        rank = None
        for pos, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                rank = pos + 1
                break
        mrr_scores.append(1.0 / rank if rank else 0.0)

    p50_latency = np.median(latencies)

    return {
        "P@10": np.mean(p10_scores),
        "nDCG@10": np.mean(ndcg10_scores),
        "R@10": np.mean(r10_scores),
        "MRR": np.mean(mrr_scores),
        "p50_latency_ms": p50_latency
    }

if __name__ == "__main__":
    print("Initializing engine and indices...")
    pipeline = RAGPipeline()

    # --- BUILD QUERIES FROM ACTUAL DATASET ---
    dataset = load_dataset("fancyzhx/ag_news", split=f"train[:{SystemConfig.DATASET_SLICE}]")
    labels = dataset['label']
    corpus = dataset['text']

    test_queries = {}
    qrels = {}
    qrel_grades = {}
    query_indices = []

    GT_SIZE = 50
    for label_id, class_name in enumerate(["World", "Sports", "Business", "Sci/Tech"]):
        class_indices = [i for i, lbl in enumerate(labels) if lbl == label_id]
        for j in range(2):
            query_pos = j * 100
            doc_idx = class_indices[query_pos]
            qid = f"q_{class_name.lower()}_{j}"
            query_text = corpus[doc_idx][:80].rsplit(' ', 1)[0]
            test_queries[qid] = query_text
            query_indices.append(doc_idx)

    # Ground truth: class-label relevance with graded similarity scores
    print("Encoding corpus for ground truth computation...")
    corpus_embs = pipeline.embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(corpus_embs)

    label_to_indices = {}
    for idx, lbl in enumerate(labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    for i, (qid, _) in enumerate(test_queries.items()):
        q_vec = pipeline.embedder.encode([test_queries[qid]], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        q_label = labels[query_indices[i]]

        same_class_dotprods = []
        q_emb_flat = q_vec[0]
        for idx in label_to_indices[q_label]:
            if idx != query_indices[i]:
                same_class_dotprods.append((idx, float(corpus_embs[idx] @ q_emb_flat)))

        same_class_dotprods.sort(key=lambda x: x[1], reverse=True)
        top = same_class_dotprods[:GT_SIZE]

        qrels[qid] = {doc_id for doc_id, _ in top}
        qrel_grades[qid] = {doc_id: sim for doc_id, sim in top}

    del corpus_embs

    # Record Baseline RAM
    base_ram = get_process_memory_mb()

    print("\nRunning Evaluation on standard FP32 Index...")
    fp32_metrics = run_evaluation(pipeline, test_queries, qrels, qrel_grades, use_quantization=False)
    fp32_ram = get_process_memory_mb() - base_ram

    print("Running Evaluation on quantized INT8 Index...")
    int8_metrics = run_evaluation(pipeline, test_queries, qrels, qrel_grades, use_quantization=True)
    int8_ram = get_process_memory_mb() - base_ram # Approximation of delta footprint

    # --- SAVE TO A SEPARATE TXT FILE ---
    report_path = "logs/retrieval_benchmark.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("                      RETRIEVAL PIPELINE BENCHMARK\n")
        f.write("========================================================================\n\n")

        f.write(f"Dataset Size Evaluated: {len(pipeline.corpus)} records\n")
        f.write(f"Test queries processed: {len(test_queries)}\n\n")

        f.write(f"{'Index Variant':<20} | {'P@10':<8} | {'nDCG@10':<10} | {'R@10':<8} | {'MRR':<8} | {'p50 Latency':<12} | {'Est. RAM'}\n")
        f.write("-" * 90 + "\n")

        f.write(f"{'IVF (FP32)':<20} | "
                f"{fp32_metrics['P@10']:<8.4f} | "
                f"{fp32_metrics['nDCG@10']:<10.4f} | "
                f"{fp32_metrics['R@10']:<8.4f} | "
                f"{fp32_metrics['MRR']:<8.4f} | "
                f"{fp32_metrics['p50_latency_ms']:<9.2f} ms | "
                f"{fp32_ram:.2f} MB\n")

        f.write(f"{'IVF-SQ (INT8)':<20} | "
                f"{int8_metrics['P@10']:<8.4f} | "
                f"{int8_metrics['nDCG@10']:<10.4f} | "
                f"{int8_metrics['R@10']:<8.4f} | "
                f"{int8_metrics['MRR']:<8.4f} | "
                f"{int8_metrics['p50_latency_ms']:<9.2f} ms | "
                f"{int8_ram:.2f} MB\n")
        f.write("========================================================================\n")

    print(f"\n[SUCCESS] Retrieval metrics written to {report_path}")