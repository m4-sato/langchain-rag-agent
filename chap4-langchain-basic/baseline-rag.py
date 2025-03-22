import os
from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loader.load()
print(len(raw_docs))


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(raw_docs)
print(len(docs))

# Embeddingsの定義
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),       # 旧: openai_api_base
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),             # 旧: openai_api_key
    openai_api_version="2023-05-15",                           # バージョンを合わせる
    model="text-embedding-3-small",
    chunk_size=1
    )

query = "GoogleDriveに保管されているpdfファイルを読み込むDocument loaderはありますか？"

vector = embeddings.embed_query(query)
print(len(vector))
print(vector)

db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

context_docs = retriever.invoke(query)
print(f"len = {len(context_docs)}")

first_doc = context_docs[0]
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)