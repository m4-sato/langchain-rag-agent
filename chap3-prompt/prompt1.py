import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview"
)

prompt = '''\
以下の料理のレシピを考えてください。

料理名： """
{dish}
"""
'''

def generate_recipe(dish: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.format(dish=dish)},
        ],
    )

    return response.choices[0].message.content

recipe = generate_recipe("カレー")
print(recipe)

system_prompt = """\
ユーザーが入力した料理のレシピを考えてください。
出力は以下のJSON形式にしてください。
```

{
    "料理": ["材料1", "材料2"],
    "手順": ["手順1", "手順2"]
}
```
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "カレー"}
    ],
)
print(response.choices[0].message.content)



##########################
# Zero-shotプロンプティング
##########################

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "ユーザーからの質問に100文字以内で答えて下さい。"},
        {"role": "user", "content": "プロンプトエンジニアリングとは？"}
    ],
)

print(response.choices[0].message.content)

##########################
# Few-shotプロンプティング
##########################

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "次の果物の色を教えて下さい。"},
        {"role": "user", "content": "りんごは？"},
        {"role": "assistant", "content": "赤色"},
        {"role": "user", "content": "バナナは？"},
        {"role": "assistant", "content": "黄色"},
        {"role": "user", "content": "レモンは？"},
    ],
)

print(response.choices[0].message.content)

###########################################
# Zero-shot Chain-of-Thoughtプロンプティング
###########################################

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "ステップバイステップで考えてください。"},
        {"role": "user", "content": "102*12*445*111"},
    ],
)

print(response.choices[0].message.content)