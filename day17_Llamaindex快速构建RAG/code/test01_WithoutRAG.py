from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM

#使用HuggingFaceLLM加载本地大模型,model_name和tokenizer_name都指向本地模型路径
#model_kwargs和tokenizer_kwargs都设置trust_remote_code为True，表示允许使用远程代码
llm = HuggingFaceLLM(model_name="/root/AI-WSL/models/Qwen/Qwen1.5-1.8B-Chat",
               tokenizer_name="/root/AI-WSL/models/Qwen/Qwen1.5-1.8B-Chat",
               model_kwargs={"trust_remote_code":True},
               tokenizer_kwargs={"trust_remote_code":True})
#调用模型chat引擎得到回复
rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])

print(rsp)