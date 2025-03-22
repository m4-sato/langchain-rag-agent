from typing import Iterator

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)


llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

output_parser = StrOutputParser()

def upper(input_stream: Iterator[str]) -> Iterator[str]:
    for text in input_stream:
        yield text.upper()

chain = prompt | llm | output_parser | upper

for chunk in chain.stream({"input": "AIエンジニアについて教えてください。"}):
    print(chunk, end="", flush=True)