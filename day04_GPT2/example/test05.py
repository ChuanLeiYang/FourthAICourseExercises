#中文诗词
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline
import torch#这里是为了让DEVICE可以自适应"cuda"还是"cpu"

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")
tokenizer = BertTokenizer.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")
print(model)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device=DEVICE)

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("白日依山尽，", max_length=50, do_sample=True))