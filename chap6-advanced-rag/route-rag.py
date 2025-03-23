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

langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"}
)

web_retriever = TavilySearchAPIRetriever(k=3).with_config(
    {"run_name": "web_retreiver"}
)

class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"

class RouteOutput(BaseModel):
    route: Route

route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するために適切なRetrieverを選択してください。
質問: {question}
""")

route_chain = (
    route_prompt
    | llm.with_structured_output(RouteOutput)
    | (lambda x: x.route)
)

def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)
    
    raise ValueError(f"Unknown retreiver: {retriever}")


route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt | llm | StrOutputParser()
)

route_rag_chain.invoke("LangChainの概要を教えて")