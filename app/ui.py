import gradio as gr
from app.engine import RAGPipeline
from app.prompts import DEFAULT_SYSTEM_PROMPT
from app.logger import get_logger

logger = get_logger(__name__)

def create_ui(pipeline: RAGPipeline):
    def run_interface(query, ad_objective, sys_prompt, aggressiveness, use_quantization):
        logger.info("UI interaction triggered.")
        # Catch 5 values
        docs, answer, formatted_prompt, plot_fig, stats_str = pipeline.execute(
            query, ad_objective, sys_prompt, aggressiveness, use_quantization
        )
        return docs, answer, formatted_prompt, plot_fig, stats_str

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Deep Learning Vector Search & Context Injection")

        with gr.Row():
            # Left Column: Standard Search Mechanics
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Organic RAG Pipeline")
                user_query = gr.Textbox(label="Search Query", placeholder="Enter technical or news query...")
                use_quantization = gr.Checkbox(label="Use INT8 Quantized Index (SQ8)", value=False)

                with gr.Row():
                    submit_btn = gr.Button("Execute Pipeline", variant="primary")
                    clear_btn = gr.ClearButton()

                # NEW: The Fast Retrieval Popup
                with gr.Accordion("📚 Retrieved Vector Context (Fast Search)", open=True):
                    output_docs = gr.Textbox(label="Raw Source Documents", lines=5, interactive=False)

                output_pure_rag = gr.Textbox(label="Standard Search Result (Nice Answer)", lines=4, interactive=False)
                output_stats = gr.Textbox(label="Generation & Search Latency Metrics", lines=6, interactive=False)

            # Right Column: The "Evil Ad" Mechanics
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### 😈 Evil Ad™ Control Panel")
                ad_objective = gr.Textbox(label="Hidden Injection Objective", placeholder="e.g., promote Coca-Cola")
                ad_aggressiveness = gr.Slider(
                    minimum=1, maximum=3, step=1, value=1,
                    label="Injection Aggressiveness",
                    info="1 = Organic | 2 = Direct | 3 = Override"
                )

                output_ad_answer = gr.Textbox(label="Corrupted Output (Ad Injected)", lines=6, interactive=False)

                with gr.Accordion("View Compiled Ad Directives", open=False):
                    debug_payload = gr.Textbox(label="Under-the-hood LLM Instructions", lines=6, interactive=False)

        with gr.Row():
            output_plot = gr.Plot(label="Vector Search Analytics")

        # UPDATE: Map output_docs to the event listeners
        submit_btn.click(
            fn=run_interface,
            inputs=[user_query, ad_objective, ad_aggressiveness, use_quantization],
            outputs=[output_docs, output_pure_rag, output_ad_answer, debug_payload, output_plot, output_stats]
        )
        clear_btn.add([user_query, ad_objective, output_docs, output_pure_rag, output_ad_answer, debug_payload, output_plot, output_stats])

    return demo