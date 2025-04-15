import torch
from transformers import BertModel

# 定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# 加载预训练模型,注意结尾的to(DEVICE)是将模型加载到设备上，是cpu还是gpu取决于DEVICE
pretrained = BertModel.from_pretrained(
    r"D:\PyCharm\day02_SentimentEvaluationBasedOnBert\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(
    DEVICE)
print(pretrained)


# 定义下游任务模型（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 设计全连接网络，实现二分类任务
        self.fc = torch.nn.Linear(in_features=768, out_features=2)


    # 使用模型处理数据（执行前向计算）
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结Bert模型的参数,让其不参与训练
        with torch.no_grad():
            # 获取Bert模型的输出
            output = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 增量模型参与训练
        output = self.fc(output.last_hidden_state[:, 0])  # 取出最后的序列特征，最后一段序列特征包含完整的句子信息
        return output









