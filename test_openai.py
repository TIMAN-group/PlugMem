import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api-inference.modelscope.cn/v1",
    api_key='ms-e1e9c1f5-3b31-49e0-88b2-60dfde6c433c',
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-27B",
    messages=[{"role": "user", "content": "Type \"I love Qwen3.5\" backwards"}],
    max_tokens=64,
    temperature=0,
)

print(response.choices[0].message.content)