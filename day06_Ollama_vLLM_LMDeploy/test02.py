#使用openai的API风格调用ollama
from openai import OpenAI
#填写远程服务器的地址和API密钥，这里的api_key是公开的，可以随便写，但是要写
client = OpenAI(base_url="http://localhost:11434/v1/",api_key="suibianxie")
#创建一个会话
chat_completion = client.chat.completions.create(
    messages=[{"role":"user","content":"你好，请介绍下你自己。"}],model="qwen2.5:0.5b"
)
print(chat_completion.choices[0])