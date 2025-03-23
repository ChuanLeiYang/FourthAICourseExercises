#本案例演示在不对生成文本格式进行规定的情况下，加载训练权重生成古诗词的效果

#导入模型、分词器、pipeline
from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#加载我们自己训练的权重（中文古诗词）
model.load_state_dict(torch.load("params/net.pt"))

#使用系统自带的pipeline工具生成内容
pipeline = TextGenerationPipeline(model,tokenizer,device=0)
#生成24个字的古诗，以“天高”为开头
print(pipeline("天高",max_length =24))