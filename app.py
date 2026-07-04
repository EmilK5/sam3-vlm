"""
app.py

Gradio User Interface layout and session state execution manager 
for the deliberative training-free orchard system.
"""

import os
import sys
import torch
import logging
import gradio as gr
from PIL import Image

# Path configuration for SAM3 repository
SAM3_REPO_ROOT = "/home/ekielar/sam3"
BPE_PATH = f"{SAM3_REPO_ROOT}/assets/bpe_simple_vocab_16e6.txt.gz"

if SAM3_REPO_ROOT not in sys.path:
    sys.path.append(SAM3_REPO_ROOT)

# Ensure local module visibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import OrchardGraph
import inference
import pipeline

# ==========================================
# 1. System Initialization
# ==========================================

# Set up the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True
)

# Set up the compute device context
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info(f"Using device: {device}")

# Initial static baseline configurations
CONFIDENCE_INITIAL = 0.35

# Pre-load the global SAM3 model components
MODEL, PROCESSOR = inference.load_sam3_model(BPE_PATH, CONFIDENCE_INITIAL, device=device)

# ==========================================
# 2. UI Driver Functions
# ==========================================

def ui_loop_driver(image, conf, clahe, tiling, prompt_text, current_pass, graph_state):
    """
    Triggered whenever the user hits the 'Run Next Pass' execution button.
    Maintains persistent memory registers across infinite sequence passes,
    allowing live runtime adjustment of prompt terminology and spatial engines.
    """
    if image is None:
        return None, current_pass, graph_state, "Error: Source canvas is empty. Please upload an image."

    if not prompt_text.strip():
        return None, current_pass, graph_state, "Error: Target concept prompt cannot be blank."

    # Initialize a clean tracking graph state database if it's the first execution pass
    if graph_state is None:
        graph_state = OrchardGraph()
        current_pass = 1
    else:
        current_pass += 1

    # Hand off execution control to the main loop pipeline referee with the new custom prompt string
    new_discoveries = pipeline.execute_pass(
        processor=PROCESSOR,
        image_pil=image,
        graph=graph_state,
        conf=float(conf),
        clahe=clahe,
        tiling=tiling,
        pass_number=current_pass,
        prompt=prompt_text
    )

    # Render current verification scene graph state back to a static view image
    output_filename = "session_output.jpg"
    inference.plot_graph_scene(image, graph_state, output_path=output_filename)

    status_msg = f"Pass {current_pass} complete! Target Prompt: '{prompt_text}' | Audited, verified, and integrated {new_discoveries} new components."
    return output_filename, current_pass, graph_state, status_msg


def ui_clear_driver():
    """
    Resets the persistent backend memory registries completely for a fresh image run.
    """
    new_graph = OrchardGraph()

    return None, 0, new_graph, "Database tables dropped. System primed for new tree canvas image."
        
# ==========================================
# 3. Gradio Interface Blocks Layout
# ==========================================

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# SAM 3 Orchard System")
    
    # Persistent State Registers kept active across click interactions
    graph_register = gr.State()
    pass_index = gr.Number(value=0, visible=True)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload or Drag Image Here",
                height=800
            )

            prompt_input = gr.Textbox(
                value="green fruit",
                label="Text Prompt",
                placeholder="Enter text prompt (e.g., green fruit, apple, orange)..."
            )

            conf = gr.Number(
                value=0.35,
                minimum=0.0,
                maximum=1.0,
                label="Confidence Threshold"
            )

            with gr.Row():
                # These settings can be toggled on/off on the fly after any pass index
                clahe = gr.Checkbox(
                    value=False,
                    label="Apply CLAHE Enhancement"
                )

                tiling = gr.Checkbox(
                    value=False,
                    label="Apply Tiling"
                )
            
            with gr.Row():
                submit_btn = gr.Button("Run Next Pass", variant="primary")
                clear_btn = gr.Button("Clear Session State", variant="secondary")

            ui_log = gr.Textbox(
                label="Output Logs", 
                value="Primes initialized. System Ready."
            )

        with gr.Column():
            image_output = gr.Image(
                label="Segmentation Result",
                height=800
            )

    # Event Mapping Hooks
    image_input.change(
        fn=ui_clear_driver,
        outputs=[image_output, pass_index, graph_register, ui_log]
    )

    submit_btn.click(
        fn=ui_loop_driver,
        inputs=[image_input, conf, clahe, tiling, prompt_input, pass_index, graph_register],
        outputs=[image_output, pass_index, graph_register, ui_log]
    )

    clear_btn.click(
        fn=ui_clear_driver,
        outputs=[image_output, pass_index, graph_register, ui_log]
    )

# ==========================================
# 4. Entry Execution Point
# ==========================================

def main():
    logging.info(f"System fully booted using compute environment resource context: {device}")
    demo.launch()

if __name__ == "__main__":
    main()