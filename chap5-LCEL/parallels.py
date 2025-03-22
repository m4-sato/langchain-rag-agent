import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import pprint
from langchain_core.runnables import RunnableParallel

load_dotenv()

optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。"),
        ("human", "{topic}"),
    ]
)

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。"),
        ("human", "{topic}"),
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

optimistic_chain = optimistic_prompt | llm | output_parser
pessimistic_chain = pessimistic_prompt | llm | output_parser

parallel_chain = RunnableParallel(
    {
        "optimistic_option": optimistic_chain,
        "pessimistic_option": pessimistic_chain,
    }
)

output = parallel_chain.invoke({"topic": "生成AIの進化について"})
pprint.pprint(output)