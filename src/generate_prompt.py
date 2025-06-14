from langchain_core.prompts import ChatPromptTemplate

def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """You are an AI assistant. Using ONLY the following context, answer the user's question as briefly and accurately as possible.
{context}

Question: {question}

Answer:"""


prompt = ChatPromptTemplate.from_template(template)