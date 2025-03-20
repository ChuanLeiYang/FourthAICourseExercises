from torch.utils.data import Dataset
from datasets import load_dataset

class MyDataset(Dataset):
    def __init__(self,split):
        #从磁盘加载csv数据
        self.dataset = load_dataset(path="csv",data_files=f"D:/AI/data/Weibo/{split}.csv",split="train")

    #返回数据集长度
    def __len__(self):
        return len(self.dataset)

    #对每条数据单独处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text,label

if __name__ == '__main__':
    dataset = MyDataset("test")
    for data in dataset:
        print(data)