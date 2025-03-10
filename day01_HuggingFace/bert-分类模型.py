#BertForSequenceClassification是Bert做语句分类的模型，BertTokenizer是Bert的分词器
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

#将模型和分词器下载到本地，并指定保存路径,第一次从网上下载用相对路径，如果模型已经下载完成用绝对路径就会用本地的模型
#model_dir=r"model\bert-base-chinese"#从网上下
model_dir=r"D:\PyCharm\day01_HuggingFace\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"#加载本地的

#加载模型和分词器,采用这种方式加载模型不管本地是否有模型都会先访问谷歌云盘，然后验证本地是否有模型如果本地有则用本地的，如果没有则从网上下
model=BertForSequenceClassification.from_pretrained("bert-base-chinese",cache_dir=model_dir)
tokenizer=BertTokenizer.from_pretrained("bert-base-chinese",cache_dir=model_dir)

#创建分类pipeline模型device="cuda"获取“cpu
classifier=pipeline("text-classification",model=model,tokenizer=tokenizer,device="cuda")

#进行文本分类
result=classifier("你好，我是一款语言模型")
print(result)
print(model)