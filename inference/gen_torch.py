import sys
import fire
import torch
import transformers
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main(
    load_8bit: bool = False,
    base_model: str = "/mnt/disk1/lxl/huggingface/openllama-english-7b-evol-intruct"
):
    evol_intruct = "evol" in base_model
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half() 

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input,evol_intruct)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    gr.Interface(
        fn=evaluate,
        inputs=[
        gr.components.Textbox(
            lines=2, label="Instruction", placeholder="Tell me about alpacas."
        ),
        gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, label="Top k"
        ),
        gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        ),
    ]
,
        outputs=[
            gr.inputs.Textbox(
                lines=30,
                label="Output",
            )
        ],
        title="OpenLLaMA-Chinese",
        description="Improve OpenLLaMA model to follow chinese instructions. If your instruction does not require any input, please leave the input field empty.",
    ).launch(share=True)


def generate_prompt(instruction, input=None,evol_intruct = False):
    if evol_intruct:
        return f"{instruction}\n\n### Response:"

    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
   fire.Fire(main)