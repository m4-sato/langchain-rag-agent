import os
from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.retrievers import BM25Retriever
from pydantic import BaseModel, Field
from enum import Enum
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

retriever = db.as_retriever()

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

def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60,
) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            content_score_mapping[content] += 1 / (rank + k)
    
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]

chroma_retriever = retriever.with_config(
    {"run_name": "chroma_retreiver"}
)

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retreiver"}
)

hybrid_retriever = (
    RunnableParallel({
        "chroma_documents": chroma_retriever,
        "bm25_documents": bm25_retriever,
    })
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt | llm | StrOutputParser()
)

hybrid_rag_chain.invoke("LangChainの概要を教えて")