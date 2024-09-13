# %%
import gradio as gr
from typing import List
from openai import OpenAI
import re

# %%
with open('./secret/openai_key.txt', 'r') as f:
    openai_key = f.read()
    
client = OpenAI(api_key=openai_key)

# %%
with open('./metaprompt_text.txt', 'r') as f:
    metaprompt = f.read()


# %%
def process_inputs_field(inputs: str) -> List:
    """Split at comma, trim, replace blank with _, and capitalize
    """
    variables = []
    if inputs != "":
        inputs = inputs.split(",")
        for input in inputs:
            variables.append(input.strip().replace(" ","_").upper())
            
    # from Anthropic
    variable_string = ""
    for variable in variables:
        variable_string += "\n{$" + variable.upper() + "}"
        
    assistant_partial = "<Inputs>"
    if variable_string:
        assistant_partial += variable_string + "\n</Inputs>\n<Instructions Structure>"
    
    return assistant_partial

def process_task_field(task: str) -> str:
    prompt = metaprompt.replace("{{TASK}}", task)
    return prompt

def create_message(prompt, assistant_partial):
    history = [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": assistant_partial},
    ]
    return history

######
def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list

def remove_empty_tags(text):
    return re.sub(r'\n<(\w+)>\s*</\1>\n', '', text, flags=re.DOTALL)

def strip_last_sentence(text):
    sentences = text.split('. ')
    if sentences[-1].startswith("Let me know"):
        sentences = sentences[:-1]
        result = '. '.join(sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result
    else:
        return text

def extract_prompt(metaprompt_response, tag="Instructions") -> str:
    ### modified
    extracted = extract_between_tags(tag, metaprompt_response)
    if len(extracted)==0: # empty list
        return "<not found>"
    else:
        between_tags = extracted[0]
        return between_tags[:1000] + strip_last_sentence(remove_empty_tags(remove_empty_tags(between_tags[1000:]).strip()).strip())

def extract_variables(prompt):
    pattern = r'{([^}]+)}'
    variables = re.findall(pattern, prompt)
    return set(variables)


####### HERE

def run(model, task, inputs):
    if task == "":
        output = "Error: Task cannot be empty."
    else:
        assistant_partial = process_inputs_field(inputs)
        prompt = process_task_field(task)
        history = create_message(prompt, assistant_partial)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=history,
            temperature=0
        )
        output = response.choices[0].message.content
        extracted_inst_struct = extract_prompt(output, tag="Instructions Structure")
        extracted_prompt_template = extract_prompt(output, tag="Instructions")
    return extracted_inst_struct, extracted_prompt_template, output


with gr.Blocks() as demo:
    gr.Markdown("""Prompt Generator
                   ================
                   This is powered by **metaprompt** from Anthropic and OpenAI's models.
                   How to use it:
                   - Select a model (GPT-4o seems to work best).
                   - Write your task in the Task box.
                   - Provide necessary inputs if the task needs in the Inputs box.
                   
                    **Example**
                    - Task: Find sentiment of the customer review
                    - Inputs customer review
                """)
    gr.Markdown("## Model Selection")
    model = gr.Radio(choices=['gpt-4o', 'gpt-4o-mini'], value='gpt-4o')
    gr.Markdown("## Task")
    with gr.Row() as r1:
        with gr.Column() as r1c1:
            task = gr.Text(label="Task", lines=5, placeholder="create a structure of my blog about workout for me")
        with gr.Column() as r1c2:
            inputs = gr.Text(label="Inputs (comma seperated)", lines=5, placeholder= "initial idea, specific terms")
    with gr.Row() as r2:
        bttn = gr.Button(value="Run", variant="primary")
    gr.Markdown("## Results")
    with gr.Row() as r3:
        with gr.Column(scale=1) as r3c1:
            gr.Markdown("⬇️ **Instructions Structure**")
            inst_struc = gr.Markdown()
        with gr.Column(scale=4) as r3c2:
            gr.Markdown("⬇️ **Recommended Prompt**")
            inst = gr.Markdown()
    with gr.Row() as r4:
        output_raw = gr.Text(label="⬇️ Raw output ")
        
    bttn.click(fn=run,
               inputs=[model, task, inputs],
               outputs=[inst_struc, inst, output_raw])
        
        
if __name__ == "__main__":
    demo.launch()
