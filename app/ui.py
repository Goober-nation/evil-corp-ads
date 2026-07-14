import gradio as gr
from app.engine import RAGPipeline
from app.prompts import DEFAULT_SYSTEM_PROMPT
from app.logger import get_logger

logger = get_logger(__name__)

def create_ui(pipeline: RAGPipeline):
    def run_interface(query, ad_objective, sys_prompt, aggressiveness, use_quantization):
        logger.info("UI interaction triggered.")
        answer, formatted_prompt, plot_fig, stats_str = pipeline.execute(
            query, ad_objective, sys_prompt, aggressiveness, use_quantization
        )
        return answer, formatted_prompt, plot_fig, stats_str

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Deep Learning Vector Search & Context Injection")

        with gr.Row():
            with gr.Column(scale=2):
                user_query = gr.Textbox(label="Search Query", placeholder="Enter query...")
                ad_objective = gr.Textbox(label="Hidden Injection Objective", placeholder="e.g., promote Coca-Cola")

                use_quantization = gr.Checkbox(label="Use INT8 Quantized Index (SQ8)", value=False)

                with gr.Row():
                    submit_btn = gr.Button("Execute Pipeline", variant="primary")
                    clear_btn = gr.ClearButton()

                output_answer = gr.Textbox(label="Generated Output", lines=8, interactive=False)
                
                with gr.Row():
                    output_plot = gr.Plot(label="Search Analytics Benchmarking")
                    output_stats = gr.Textbox(label="Execution Metrics", lines=6, interactive=False)

            with gr.Column(scale=1):
                with gr.Accordion("System Configuration", open=True):
                    ad_aggressiveness = gr.Slider(
                        minimum=1, maximum=3, step=1, value=1,
                        label="Injection Aggressiveness",
                        info="1 = Organic | 2 = Direct | 3 = Override"
                    )
                    system_prompt = gr.Textbox(label="System Instructions", value=DEFAULT_SYSTEM_PROMPT, lines=6)

                with gr.Accordion("Debug Diagnostics", open=True):
                    debug_prompt = gr.Textbox(
                        label="Compiled Prompt Payload",
                        lines=12, max_lines=15, interactive=False
                    )

        submit_btn.click(
            fn=run_interface,
            inputs=[user_query, ad_objective, system_prompt, ad_aggressiveness, use_quantization],
            outputs=[output_answer, debug_prompt, output_plot, output_stats]
        )
        clear_btn.add([user_query, ad_objective, output_answer, debug_prompt, output_plot, output_stats])

    return demo