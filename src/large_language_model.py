from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline

import os
import yaml
from pathlib import Path
project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)

with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

def load_llm(params):
    llm_model_name = params["model"]["llm_model"]
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=float(params["model"]["temperature"]),
        top_k=int(params["model"]["top_k"]),
        top_p=float(params["model"]["temperature"]),
        repetition_penalty=float(params["model"]["temperature"]),
        do_sample=True,
        )
    
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return llm

llm = load_llm(params)
r = llm.invoke("How many Holidays do we get?")
print(r)