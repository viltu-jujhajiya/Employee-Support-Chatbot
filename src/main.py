from generate_prompt import doc2str, prompt
from large_language_model import load_llm
import os
import yaml
from pathlib import Path
import torch
from langchain.chains import retrieval_qa
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)

with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

embedding_model_name = params["data"]["embedding_model"]
vectordb_path = Path(os.path.join(params["data"]["vectordb_dirpath"], params["data"]["vector_db_name"]))

def load_vectorstore():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name = embedding_model_name,
        model_kwargs = {"device": device},
        encode_kwargs = {"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(vectordb_path, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore

def main():
    '''Load Vector Store & set-up Retriever'''
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    
    '''Load LLM'''
    llm = load_llm(params)
    
    '''Create Langchain '''
    rag_chain = (
        {
            "context": retriever | doc2str,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(rag_chain.invoke("How many Holidays do we get?"))

if __name__ == "__main__":
    main()