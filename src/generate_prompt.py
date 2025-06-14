from langchain_core.prompts import ChatPromptTemplate

def doc2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Answer the question based on the following context:
{context}

Question: {question}

Answer: """


prompt = ChatPromptTemplate.from_template(template)