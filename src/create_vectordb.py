from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import os
import re
import yaml
from pathlib import Path


project_path = Path(__file__).resolve().parent.parent
os.chdir(project_path)

with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.replace('\u200b', '')
    return text.strip()

datapath = params["data"]["datapath"]

pdf_loader = DirectoryLoader(
    datapath,
    glob = "**/*.pdf",
    loader_cls = PyMuPDFLoader,
    show_progress = True,
)

docx_loader = DirectoryLoader(
    datapath,
    glob = "**/*.docx",
    loader_cls = UnstructuredWordDocumentLoader,
    show_progress = True,
)


docs = pdf_loader.load() + docx_loader.load()

data = []
for doc in docs:
    doc.page_content = clean_text(doc.page_content)
    if doc.page_content and len(doc.page_content.strip()) > 100:
        data.append(doc)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    separators=["\n\n", "\n", ".", " ", ""],
)

data_chunks = text_splitter.split_documents(data)
print(len(data_chunks))

embedding_model_name = params["data"]["embedding_model"]
vectordb_path = Path(os.path.join(params["data"]["vectordb_dirpath"], params["data"]["vector_db_name"]))

device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = HuggingFaceBgeEmbeddings(
    model_name = embedding_model_name,
    model_kwargs = {"device": device},
    encode_kwargs = {"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(data_chunks, embedding_model)
vectorstore.save_local(vectordb_path)