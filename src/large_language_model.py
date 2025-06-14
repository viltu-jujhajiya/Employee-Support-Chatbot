from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline

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
        do_sample=params["model"]["do_sample"],
        )
    
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return llm