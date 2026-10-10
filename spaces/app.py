"""MEGA demo Space.

A tiny Gradio app over the published ``Frankenstein-Labs/mega-small`` model.

Be honest in the UI: this is a small research model trained on a structured
synthetic corpus. It completes arithmetic and templated sentences from that
corpus and nothing else. The banner says so, because a demo that oversells a
model is worse than no demo.
"""

from __future__ import annotations

import os

import gradio as gr
import torch

from mega.hub.serialization import from_pretrained, load_generation_config
from mega.tokenizer.tokenizer import MegaTokenizer

MODEL_ID = os.environ.get("MEGA_MODEL_ID", "Frankenstein-Labs/mega-small")

DISCLAIMER = (
    "**Research demo.** This is a small Mixture-of-Experts model trained from "
    "scratch on a tiny structured corpus (arithmetic, templated fr/en sentences, "
    "small code snippets). It can complete those patterns and little else. It is "
    "not an assistant, has no safety tuning, and will produce nonsense on general "
    "questions. Source: [mega-moe](https://github.com/Frankenstein-Labs/mega-moe)."
)


def load():
    model = from_pretrained(MODEL_ID)
    tok = MegaTokenizer.from_pretrained(MODEL_ID)
    gen = load_generation_config(MODEL_ID) if os.path.isdir(MODEL_ID) else {}
    model.eval()
    return model, tok, gen


MODEL, TOK, GEN = load()


def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int, seed: int) -> str:
    if not prompt.strip():
        return "(enter a prompt)"
    torch.manual_seed(int(seed))
    ids = TOK.encode(prompt, add_bos=True)
    out = MODEL.generate(
        torch.tensor([ids]),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k) or None,
        top_p=None if temperature <= 0 else GEN.get("top_p"),
        eos_token_id=TOK.eos_token_id,
    )
    continuation = TOK.decode(out[0, len(ids):].tolist())
    return f"{prompt}{continuation}"


EXAMPLES = [
    ["Le chat noir mange une pomme .", 12, 0.0, 40, 0],
    ["Q: Calcule : 39 + 62 A:", 8, 0.0, 40, 0],
    ["Q: What is 37 - 18? A:", 8, 0.0, 40, 0],
    ["73 - 42 =", 8, 0.0, 40, 0],
    ["def add(a, b): return a + b  #", 16, 0.0, 40, 0],
    ["Alice a 7 billes et Bob a 5 billes . Total :", 8, 0.0, 40, 0],
]

with gr.Blocks(title="MEGA — MoE reference model") as demo:
    gr.Markdown("# MEGA — reference MoE language model")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, value=EXAMPLES[0][0])
            with gr.Row():
                max_new = gr.Slider(1, 64, value=16, step=1, label="max new tokens")
                temp = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="temperature (0 = greedy)")
            with gr.Row():
                top_k = gr.Slider(0, 200, value=40, step=1, label="top-k")
                seed = gr.Number(value=0, label="seed", precision=0)
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Output", lines=10)
    gr.Examples(examples=EXAMPLES, inputs=[prompt, max_new, temp, top_k, seed])
    btn.click(generate, [prompt, max_new, temp, top_k, seed], out)
    prompt.submit(generate, [prompt, max_new, temp, top_k, seed], out)

if __name__ == "__main__":
    demo.launch()
