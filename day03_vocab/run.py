#模型使用接口
#导入模型用
import torch
#用于加载刚刚设计的模型
from net import Model
#用于加载预训练模型,BertTokenizer是用于加载字典和分词器，AdamW是用于加载优化器
from transformers import BertTokenizer

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加载字典和分词器
token=BertTokenizer.from_pretrained(r"D:\AI\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
model=Model().to(DEVICE)
names= [
                "like",
                "disgust",
                "happiness",
                "sadness",
                "anger",
                "surprise",
                "fear",
                "none"
            ]

#将传进来的字符串数据进行编码
def collate_fn(data):
    sents=[]
    sents.append(data)
    #批量编码句子
    data=token.batch_encode_plus(
        batch_text_or_text_pairs=sents,#传入句子
        truncation=True,#是否截断，当句子长度大于max_length是就截断
        max_length=512,#max_length的上限是512，!!!!注意，测试时候的这长度一定要和训练时候保持一直
        padding="max_length",#如果没达到则一律补0到max_length
        return_tensors="pt",#编码后返回数值的类型，可取值为tf,pt,np,给None默认为list
        return_length=True#返回序列长度
    )

    input_ids=data["input_ids"]
    attention_mask=data["attention_mask"]
    token_type_ids=data["token_type_ids"]
    return input_ids,attention_mask,token_type_ids

def test():
    #加载模型训练参数
    model.load_state_dict(torch.load("ParamsSave/last_bert.pth"))
    #开启测试模式
    model.eval()
    while True:
        data=input("请输入测试数据（输入‘q'退出）：")
        if data=="q":
            print("测试结束")
            break
        input_ids,attention_mask,token_type_ids=collate_fn(data)
        input_ids=input_ids.to(DEVICE)
        attention_mask=attention_mask.to(DEVICE)
        token_type_ids=token_type_ids.to(DEVICE)

        #将输入给到模型，得到输出
        with torch.no_grad():
            out=model(input_ids,attention_mask,token_type_ids)
            out=out.argmax(dim=1)
            print("模型判定：",names[out],"\n")


if __name__=="__main__":
    test()




