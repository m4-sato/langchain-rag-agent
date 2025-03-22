import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
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

prompt_value = prompt.invoke({"dish": "カレー"})
ai_message = llm.invoke(prompt_value)
output = output_parser.invoke(ai_message)

print(output)

# invokeメソッド
chain = prompt | llm | output_parser
output = chain.invoke({"dish": "カレー"})

# streamメソッド
chain = prompt | llm | output_parser
for chunk in chain.stream({"dish": "カレー"}):
    print(chunk, end="", flush=True)

# batchメソッド
chain = prompt | llm | output_parser
outputs = chain.batch([{"dish": "カレー"}, {"dish": "うどん"}])
print(outputs)