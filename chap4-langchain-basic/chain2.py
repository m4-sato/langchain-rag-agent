import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

output_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。\n\n{format_instructions}"),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions = output_parser.get_format_instructions()
)

llm = AzureChatOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0
    ).bind(response_format={"type": "json_object"})

chain = prompt_with_format_instructions | llm | output_parser

recipe = chain.invoke({"dish": "カレー"})
print(type(recipe))
print(recipe)