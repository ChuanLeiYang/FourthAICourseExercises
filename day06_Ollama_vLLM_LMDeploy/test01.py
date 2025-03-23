#使用transformer加载qwen模型，AutoModelForCausalLM,AutoTokenizer可以自动适应模型和分词器
from transformers import AutoModelForCausalLM,AutoTokenizer

DEVICE = "cuda"

#加载本地模型路径为该模型配置文件所在的根目录
model_dir = "/home/aron/llm/Qwen/Qwen2.5-0.5B-Instruct"

#使用transformer加载模型，device_map指的是config.json里面的  "torch_dtype"这里是 "bfloat16",也可以用"auto"
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype="auto",device_map="auto")
#加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#调用模型
#定义提示词
prompt = "你好，请介绍下你自己。"
#将提示词封装为message
message = [{"role":"system","content":"You are a helpful assistant system"},{"role":"user","content":prompt}]
#使用分词器的apply_chat_template()方法将上面定义的消息列表进行转换，即将字符转换为input_ids;
# tokenize=False表示此时不进行令牌化,即暂时不将文本字符转换为input_ids,下一步会进行令牌化
text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

#将处理后的文本令牌化并转换为模型的输入张量，"pt"表示返回的张量是pytorch张量
model_inputs = tokenizer([text],return_tensors="pt").to(DEVICE)

#将数据输入模型得到输出
response = model.generate(model_inputs.input_ids,max_new_tokens=512)
print(response)

#对输出的内容进行解码还原，batch_decode批量解码，skip_special_tokens=True表示跳过特殊令牌（即不带特殊符号）
response = tokenizer.batch_decode(response,skip_special_tokens=True)
print(response)