#中文文言文生成

import torch#这里是为了让DEVICE可以自适应"cuda"还是"cpu"
from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")
tokenizer = BertTokenizer.from_pretrained(r"D:\AI\model\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")
print(model)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device=DEVICE)

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("于是者", max_length=100, do_sample=True))