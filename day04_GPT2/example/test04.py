#中文对联
import torch#这里是为了让DEVICE可以自适应"cuda"还是"cpu"
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-couplet\snapshots\91b9465fb1be617f69c6f003b0bd6e6642537bec")
tokenizer = BertTokenizer.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-couplet\snapshots\91b9465fb1be617f69c6f003b0bd6e6642537bec")
print(model)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device=DEVICE)

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("[CLS]十口心思，思君思国思社稷", max_length=28, do_sample=True))