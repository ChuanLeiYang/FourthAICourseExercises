#导入模型用
import torch

from MyData import MyDataset
#用于加载数据集
from torch.utils.data import DataLoader
#用于加载刚刚设计的模型
from net import Model
#用于加载预训练模型,BertTokenizer是用于加载字典和分词器，AdamW是用于加载优化器
from transformers import BertTokenizer
from torch.optim import AdamW #AdamW是用于加载优化器

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加载字典和分词器
token=BertTokenizer.from_pretrained(r"D:\AI\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传进来的字符串数据进行编码
def collate_fn(data):
    sents=[i[0] for i in data]
    labels=[i[1] for i in data]
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
    labels=torch.LongTensor(labels)
    return input_ids,attention_mask,token_type_ids,labels


#创建数据集
test_dataset=MyDataset("test")
#取数据
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=100,#批次要和训练时候保持一致，甚至要尽量保持训练的设备和测试的设备是同一个
    shuffle=True,#是否打乱数据
    drop_last=True,#是否丢弃最后一个不足batch_size的batch
    collate_fn=collate_fn#是否自定义数据集的处理方式
    )


if __name__=="__main__":

    acc=0.0#精度
    total=0#数据总的数量

    #开始训练
    print(DEVICE)
    model=Model().to(DEVICE)
    #加载模型训练参数
    model.load_state_dict(torch.load("params/148_bert.pth"))
    #开启测试模式
    model.eval()

    for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(test_loader):
        #将数据加载到设备上
        input_ids=input_ids.to(DEVICE)
        attention_mask=attention_mask.to(DEVICE)
        token_type_ids=token_type_ids.to(DEVICE)
        labels=labels.to(DEVICE)

        #前向计算(将数据输入模型，得到输出)
        out=model(input_ids,attention_mask,token_type_ids)

        out = out.argmax(dim=1)#将结果中的两个小数取概率较大的index，目的是转换成和label一样的形式

        acc+=(out==labels).sum().item()#判断输出和测的标签是否相同，相同则进行累加
        print(i,(out==labels).sum().item())
        total+=len(labels)
    print(f"test acc:{acc/total:.4f}")





