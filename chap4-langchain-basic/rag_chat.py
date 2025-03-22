import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutParser
from langchain_core.runnables import RunnablePassthrough


prompt = ChatPromptTemplate.from_templete('''\
以下の文脈だけを踏まえて質問に回答してください。'
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

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutParser()
)

output = chain.invoke(query)
print(output)