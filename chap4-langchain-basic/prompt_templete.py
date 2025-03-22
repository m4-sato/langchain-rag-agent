from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langsmith import Client


#################
# PromptTemplete
#################

prompt = PromptTemplate.from_template("""以下の料理のレシピを考えてください。
料理名：{dish}""")

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)

####################
# ChatPromptTemplete
####################

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)

#####################
# MessagesPlaceholder
#####################

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)

prompt_value = prompt.invoke(
    {
        "chat_history": [
            HumanMessage(content="こんにちは！私はジョンといいます！"),
            AIMessage(content="こんにちは！ジョンさん！どのようにお手伝いできますか？"),
        ],
        "input": "私の名前は分かりますか？"})

print(prompt_value)


client = Client()
prompt = client.pull_prompt("test")

prompt_value = prompt.invoke({"dish": "カレー"})
print(prompt_value)