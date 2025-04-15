#加载字典和分词器
from transformers import BertTokenizer
token=BertTokenizer.from_pretrained(r"D:\PyCharm\day02_SentimentEvaluationBasedOnBert\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
#print(token)

#准备要编码的文本数据
sents = ["白日依山尽，",
         "价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所"]
#批量编码句子
out=token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0],sents[1]],#传入句子
    add_special_tokens=True,#是否添加特殊字符
    truncation=True,#是否截断，当句子长度大于max_length是就截断
    max_length=8,#max_length的上限是512，
    padding="max_length",#如果没达到则一律补0到max_length
    return_tensors=None,#编码后返回数值的类型，可取值为tf,pt,np,给None默认为list
    return_attention_mask=True,
    return_token_type_ids=True,
    return_speaker_ids=True,
    return_length=True#返回序列长度
)

for k,v in out.items():
    print(k,":",v)

#解码文本数据
print(token.decode(out["input_ids"][0]),token.decode(out["input_ids"][1]))

