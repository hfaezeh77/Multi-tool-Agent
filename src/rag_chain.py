from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

load_dotenv()

PERSIST_DIR = "chroma_db"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def build_rag_chain():
    retriever = get_retriever()
    llm = ChatOpenAI(
        model="openrouter/free",   # auto-selects an available free model
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using the document context below AND any relevant facts from the conversation history if provided.
    If the answer isn't in either the context or the conversation, say you don't know — do not make anything up.

    Document context:
    {context}

    Question (may include prior conversation): {question}

    Answer:"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_rag_chain()
    while True:
        q = input("\nAsk something (or 'quit'): ")
        if q.lower() == "quit":
            break
        print(chain.invoke(q))