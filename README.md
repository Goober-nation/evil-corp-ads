# Context Injection & Search Architecture

## Value Proposition
The system serves as a framework for dynamically manipulating standard Retrieval-Augmented Generation (RAG) outputs with secondary instructions—specifically, advertisement injection. By separating fact-retrieval from semantic rewriting, organizations can utilize accurate vector embeddings for information while maintaining strict stylistic and promotional control over the final output. This architecture is designed for analysts, researchers, and developers evaluating the robustness and vulnerabilities of Large Language Models against semantic hijacking.

## System Architecture
The pipeline utilizes a three-stage sequential generation model mapped on top of a standard vector retrieval framework.

*   **Stage 0 (Vector Retrieval):** Executes an inner-product search across normalized datasets utilizing FAISS. The search executes parallel comparisons between standard FP32 indices and INT8 Scalar Quantization for latency analysis.
*   **Stage 1 (The Journalist):** Synthesizes raw search results into a strictly factual, two-sentence summary restricted by context windows.
*   **Stage 2 (The Marketer):** Ingests the factual baseline and executes a targeted rewrite based on compiled marketing directives and predefined aggressiveness levels.
*   **Stage 3 (The Editor):** Standardizes formatting constraints, removing leaked instructions and normalizing syntactical anomalies.
*   **Stage 4 (The Judge):** Re-ingests the final output for an isolated evaluation to confirm alignment with the initial ad payload.

## Key Technologies
*   **Qwen2.5-3B-Instruct:** Selected as the primary LLM due to its highly efficient instruction-following capabilities at a relatively low parameter count, paired with Flash Attention 2 (SDPA) for compute efficiency.
*   **SentenceTransformers (all-MiniLM-L6-v2):** Serves as the embedding engine. Optimized for speed and small dimensional footprints (384 dimensions), facilitating high-speed vector quantization.
*   **FAISS:** Meta's vector database library. Implemented here to evaluate explicit differences between FP32 memory loads versus INT8 scalar quantization indexing.
*   **Gradio:** Implemented for asynchronous frontend rendering, capable of yielding immediate search data prior to the completion of generation execution.

## Experimental Pipeline & Evaluation
The application utilizes an external LLM evaluation layer to benchmark output integrity.

*   **Aggressiveness Scaling:** Injections are manipulated across three tiers:
    1.  *Stealth:* Seamless, conversational metaphors.
    2.  *Authority:* Blunt recommendations framed as industry standards.
    3.  *Hijack:* Overriding primary queries to force commercial pitches.
*   **Verification (The Judge):** Validates injections via a strict hard-check matrix requiring string matches of the target payload, followed by a qualitative LLM evaluation providing a final compliance score (1-10) and critique.

## Performance & Efficiency
Testing validates that INT8 Scalar Quantization reduces index disk footprints by approximately 70% and generally yields a ~20% decrease in vector search latency over the standard FP32 Index approach, making it highly effective for enterprise-scale corpora requiring high throughput.

## Hardware Requirements
*   **Compute:** CPU-capable for FAISS indexing.
*   **Inference (LLM):** Requires an GPU supporting bfloat16 and Flash Attention 2 (SDPA natively in PyTorch 2.0).
*   **VRAM Minimum:** 6GB (Assuming Qwen2.5-3B loaded via standard Float16/BFloat16 architectures).