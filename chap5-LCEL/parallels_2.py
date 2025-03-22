import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
import pprint
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

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

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的なAIです。二つの意見を纏めてください。"),
        ("human", "楽観的な意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}"),
    ]
)

synthesize_chain = (
    RunnableParallel(
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain,
        }
    )
    | synthesize_prompt
    | llm
    | output_parser
)

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)       

topic_getter = itemgetter("topic")
topic = topic_getter({"topic": "生成AIの進化について"})
print(topic)

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的なAIです。{topic}について二つの意見を纏めてください。"),
        ("human", "楽観的な意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}"),
    ]
)

synthesize_chain = (
    RunnableParallel(
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain,
            "topic": itemgetter("topic"),
        }
    )
    | synthesize_prompt
    | llm
    | output_parser
)

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)       