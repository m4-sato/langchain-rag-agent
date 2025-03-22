import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.callbacks import LangsmithCallbackHandler

load_dotenv()

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),
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

cot_chain = cot_prompt | llm | output_parser

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけを抽出してください。"),
        ("human", "{text}"),
    ]
)

summarize_chain = summarize_prompt | llm | output_parser

cot_summarize_chain = cot_chain | summarize_chain
output = cot_summarize_chain.invoke({"question": "10+2*3"})
print(output)