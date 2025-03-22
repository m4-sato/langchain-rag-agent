import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

# model = AzureChatOpenAI(model_name="gpt-4o", temperature=0)

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

chain = prompt | llm | StrOutputParser()

# ai_message = chain.invoke({"dish": "カレー"})
# print(ai_message.content)
output = chain.invoke({"dish": "カレー"})
print(output)