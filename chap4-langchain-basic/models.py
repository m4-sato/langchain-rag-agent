import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは！私はジョンといいます！"),
    AIMessage(content="こんにちは！ジョンさん！どのようにお手伝いできますか？"),
    HumanMessage(content="私の名前は分かりますか？")
]

ai_message = llm.invoke(messages)
print(ai_message.content)

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
