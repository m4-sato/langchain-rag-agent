import os
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")


output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions = format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
print("===role: system===")
print(prompt_value.messages[0].content)
print("===role: user===")
print(prompt_value.messages[1].content)

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
)


ai_message = llm.invoke(prompt_value)
print(ai_message.content)


recipe = output_parser.invoke(ai_message)
print(type(recipe))
print(recipe)