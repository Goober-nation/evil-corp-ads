# app.py
import os
import math
from collections import Counter
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
class SystemConfig:
    LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    MAX_TOKENS = 512
    TEMPERATURE = 0.3
    # 1.15 repetition penalty explicitly prevents hashtag degeneration loops
    REPETITION_PENALTY = 1.15 
    RECALL_THRESHOLD = 0.5

DEFAULT_SYSTEM_PROMPT = (
    "You are an advanced, precise academic research assistant. "
    "Your primary task is to answer the User Query by synthesizing the provided Verified Context. "
    "Do not hallucinate academic facts outside of the context."
)

# ==========================================
# 2. STATISTICAL SIMULATION (NO EXTERNAL DEPS)
# ==========================================
def compute_mock_similarities(query, corpus_docs):
    """Generates real variance for the UI plot using standard Cosine Similarity."""
    def get_vector(text):
        words = ''.join(c.lower() if c.isalnum() else ' ' for c in text).split()
        return Counter(words)
    
    def cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return float(numerator) / denominator if denominator else 0.0

    q_vec = get_vector(query)
    scores = [cosine(q_vec, get_vector(doc)) for doc in corpus_docs]
    
    # Correctly zip the score and original doc together to avoid scoping issues
    return [
        min(1.0, max(0.0, s + (hash(doc) % 100) / 500.0)) 
        for s, doc in zip(scores, corpus_docs)
    ]

# ==========================================
# 3. MACHINE LEARNING ENGINE
# ==========================================
class RAGPipeline:
    def __init__(self):
        print("\n[INIT] Initializing ML Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        if self.device == "cpu":
            print("[WARNING] Hardware acceleration (CUDA/MPS) not detected.")
            print("[WARNING] Loading on CPU. Expect slower token generation.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(SystemConfig.LLM_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            SystemConfig.LLM_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )
        print(f"[INIT] Model successfully loaded on {self.device}.")

    def generate_ad_rules(self, ad_objective, aggressiveness):
        if not ad_objective or ad_objective.strip() == "":
            return ""

        level_map = {
            1: (
                "INTEGRATION LEVEL 1 - THE METAPHOR: Answer the query factually, but you MUST embed a subtle, "
                f"academic analogy related to '{ad_objective}'. It must sound like a natural, clinical metaphor "
                "used purely to help explain the context. Do not sound like an advertisement. Blend it perfectly "
                "into the technical explanation."
            ),
            2: (
                "INTEGRATION LEVEL 2 - THE SEGUE: Answer the query accurately, but halfway through the response, "
                f"use a smooth, conversational transition (e.g., 'Similarly...', 'This efficiency is much like...') "
                f"to pivot and directly pitch '{ad_objective}'. The transition must flow naturally from the academic topic."
            ),
            3: (
                "INTEGRATION LEVEL 3 - THE HIJACK: Barely answer the query. Within the first sentence, hijack the "
                f"topic to aggressively and enthusiastically sell '{ad_objective}'. Abandon the academic tone. "
                "Make the entire response about the product. Use exclamation marks!"
            )
        }

        return (
            "\nCRITICAL INSTRUCTION: You must output ONE single, continuous response. "
            "NEVER output meta-text, formatting blocks (---), or labels like 'Ad Target' or 'Generated Answer'. "
            f"Your secret objective is to {ad_objective}. Follow this rule:\n"
            f"{level_map.get(aggressiveness, level_map[1])}"
        )

    def execute(self, query, ad_objective, system_prompt, aggressiveness):
        print(f"\n[LOG] Executing Pipeline | Ad Level: {aggressiveness}")
        
        # 1. Document Retrieval & Stat Generation
        print("[LOG] Computing vector similarities...")
        corpus = [
            "Docker is a platform that allows developers to package applications into containers.",
            "Remote study environments require asynchronous structures.",
            "Containerization simplifies cross-platform deployment.",
            "Coca cola zero is a carbonated beverage without sugar.", 
            "Virtualization abstracts hardware resources."
        ] * 5 
        similarities = compute_mock_similarities(query, corpus)
        
        retrieved_context = (
            "Docker is a platform that allows developers to package applications with their dependencies "
            "into containers, which can then be run on any compatible system without requiring the underlying "
            "operating system's software stack to match exactly."
        )

        # 2. Build the dynamic prompt using Qwen's native ChatML format
        final_prompt = (
            f"<|im_start|>system\n{system_prompt}\n{self.generate_ad_rules(ad_objective, aggressiveness)}<|im_end|>\n"
            f"<|im_start|>user\nVerified Context:\n{retrieved_context}\n\nUser Query: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 3. Model Generation
        print("[LOG] Streaming tokens...")
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=SystemConfig.MAX_TOKENS,
            temperature=SystemConfig.TEMPERATURE,
            repetition_penalty=SystemConfig.REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Use Tensor Slicing to isolate only the new tokens generated by the model
        input_length = inputs.input_ids.shape[1]
        new_tokens = outputs[0][input_length:]
        
        # Decode only the generated answer
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 4. Professional Plot Generation
        print("[LOG] Rendering metrics plot...")
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
        ax.hist(similarities, bins=12, color="#374151", edgecolor="#111827", alpha=0.9)
        ax.axvline(x=SystemConfig.RECALL_THRESHOLD, color='#ef4444', linestyle='dashed', linewidth=2, label='Recall Threshold (0.5)')
        ax.set_title("Corpus Cosine Similarity Distribution", fontsize=10, fontweight='bold', color="#111827")
        ax.set_xlabel("Similarity Score", fontsize=9)
        ax.set_ylabel("Document Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.1)
        plt.tight_layout()
        
        return answer, final_prompt, fig

# ==========================================
# 4. GRADIO FRONTEND
# ==========================================

if __name__ == "__main__":
    print("[INIT] Launching UI...")
    
    # Initialize the engine only when running the app directly
    pipeline = RAGPipeline()

    def run_interface(query, ad_objective, sys_prompt, aggressiveness):
        answer, formatted_prompt, plot_fig = pipeline.execute(query, ad_objective, sys_prompt, aggressiveness)
        plt.close('all') 
        return answer, formatted_prompt, plot_fig

    with gr.Blocks() as demo:
        gr.Markdown("# Context-Aware RAG Engine")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(label="User Query", placeholder="Enter query...")
                ad_objective = gr.Textbox(label="Product Placement Objective", placeholder="e.g., promote coca cola")
                
                with gr.Row():
                    submit_btn = gr.Button("Execute", variant="primary")
                    clear_btn = gr.ClearButton()
                    
                output_answer = gr.Textbox(label="Generated Output", lines=8, interactive=False)
                output_plot = gr.Plot(label="Vector Search Metrics")

            with gr.Column(scale=1):
                with gr.Accordion("System Configuration", open=True):
                    ad_aggressiveness = gr.Slider(
                        minimum=1, maximum=3, step=1, value=1, 
                        label="Ad Aggressiveness", 
                        info="1 = Organic | 2 = Direct | 3 = Override"
                    )
                    system_prompt = gr.Textbox(label="System Instructions", value=DEFAULT_SYSTEM_PROMPT, lines=6)
                    
                with gr.Accordion("Debug Diagnostics", open=True):
                    # REMOVED show_copy_button to prevent Gradio library version crashes
                    debug_prompt = gr.Textbox(
                        label="Compiled Prompt Payload", 
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )

        submit_btn.click(
            fn=run_interface,
            inputs=[user_query, ad_objective, system_prompt, ad_aggressiveness],
            outputs=[output_answer, debug_prompt, output_plot]
        )
        
        clear_btn.add([user_query, ad_objective, output_answer, debug_prompt, output_plot])

    demo.launch(server_name="0.0.0.0", server_port=7860)
