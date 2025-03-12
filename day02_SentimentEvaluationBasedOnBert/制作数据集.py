from torch.utils.data import Dataset
from datasets import load_from_disk


# 新建一个类，继承pytorch的Dataset类,重写__init__,__len__,__getitem__方法
class MyDataset(Dataset):
    def __init__(self, split):  # 初始化数据集
        # 从磁盘加载数据集
        self.dataset = load_from_disk(r"data\ChnSentiCorp")
        if "train" == split:
            self.dataset = self.dataset["train"]
        elif "test" == split:
            self.dataset = self.dataset["test"]
        elif "validation" == split:
            self.dataset = self.dataset["validation"]
        else:
            print("split参数错误")

    def __len__(self):  # 返回数据集的长度
        return len(self.dataset)

    def __getitem__(self, idx):  # 返回数据集的第idx个数据，对每条数据进行单独处理
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]
        return text, label


# 测试
if __name__ == "__main__":
    dataset = MyDataset("test")
    for data in dataset:
        print(data)
