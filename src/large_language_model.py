from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_core.language_models import LLM
from pydantic import Field
from typing import List, Any

class TransformersLLM(LLM):
    generator: Any = Field()

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        result = self.generator(prompt, max_new_tokens=200)[0]["generated_text"]
        return result[len(prompt):].strip()

    @property
    def _llm_type(self) -> str:
        return "transformers-pipeline"

def load_llm(params):
    llm_model_name = params["model"]["llm_model"]
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
        temperature=float(params["model"]["temperature"]),
        )
    
    llm = TransformersLLM(generator=llm_pipeline)

    return llm