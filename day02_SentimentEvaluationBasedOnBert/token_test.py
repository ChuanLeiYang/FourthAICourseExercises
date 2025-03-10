#加载字典和分词器
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained(r"D:\PyCharm\day02_SentimentEvaluationBasedOnBert\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
#print(token)


