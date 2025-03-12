#导入模型用
import torch

from 制作数据集 import MyDataset
#用于加载数据集
from torch.utils.data import DataLoader
#用于加载刚刚设计的模型
from 模型设计 import Model
#用于加载预训练模型,BertTokenizer是用于加载字典和分词器，AdamW是用于加载优化器
from transformers import BertTokenizer,AdamW

#定义设备信息
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义训练的轮次（将整个数据集训练完一次为一个轮次）
EPOCH=30000
#加载字典和分词器
token=BertTokenizer.from_pretrained(r"D:\PyCharm\day02_SentimentEvaluationBasedOnBert\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传进来的字符串数据进行编码
def collate_fn(data):
    sents=[i[0] for i in data]
    labels=[i[1] for i in data]
    #批量编码句子
    data=token.batch_encode_plus(
        batch_text_or_text_pairs=sents,#传入句子
        truncation=True,#是否截断，当句子长度大于max_length是就截断
        max_length=512,#max_length的上限是512，
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
train_dataset=MyDataset("train")
#取数据
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=500,#每次取几条数据，超参数，让GPU能显存占用90%左右，如果cpu怎内存占用90%左右
    shuffle=True,#是否打乱数据
    drop_last=True,#是否丢弃最后一个不足batch_size的batch
    collate_fn=collate_fn#是否自定义数据集的处理方式
    )


if __name__=="__main__":
    #开始训练
    print(DEVICE)
    model=Model().to(DEVICE)
    #定义优化器
    optimizer=AdamW(model.parameters())
    #定义损失函数
    loss_func=torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            #将数据加载到设备上
            input_ids=input_ids.to(DEVICE)
            attention_mask=attention_mask.to(DEVICE)
            token_type_ids=token_type_ids.to(DEVICE)
            labels=labels.to(DEVICE)

            #前向计算(将数据输入模型，得到输出)
            output=model(input_ids,attention_mask,token_type_ids)
            #计算损失
            loss=loss_func(output,labels)
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            print(loss.item())

            #每隔5个批次输出训练信息
            if 0==i%5:
                out=out.argmax(dim=1)
                #计算训练精度
                acc=(out==labels).sum().item()/len(labels)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")

        #保存模型
        torch.save(model.state_dict(),f"params/{epoch}_bert.pth")
        print("模型已保存成功！")






