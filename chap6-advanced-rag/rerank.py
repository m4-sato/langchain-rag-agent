import os
from dotenv import load_dotenv
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Any

load_dotenv()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()
print(len(documents))

# Embeddingsの定義
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),       # 旧: openai_api_base
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),             # 旧: openai_api_key
    openai_api_version="2023-05-15",                           # バージョンを合わせる
    model="text-embedding-3-large",
    chunk_size=1
    )

db = Chroma.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈: """
{context}
"""

質問: {question}
''')

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

output_parser = StrOutputParser()

retriever = db.as_retriever()

def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    # cohere_reranker = CohereRerank(model="Cohere-rerank-v3.5", top_n=top_n)

    cohere_reranker = CohereRerank(
        model="rerank-multilingual-v3.0",   # Cohere 上で使えるリランク用モデル名
        top_n=top_n,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    return cohere_reranker.compress_documents(documents=documents, query=question)

rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank) | prompt | llm | StrOutputParser()
)

rerank_rag_chain.invoke("LangChainの概要を教えて")