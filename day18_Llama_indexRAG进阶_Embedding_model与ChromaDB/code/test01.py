#embedding_model效果对比
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

# 加载SQuAD数据（假设已处理成列表格式）
with open("squad_dev.json") as f:
    squad_data = json.load(f)["data"]
    
# 提取问题和答案对
qa_pairs = []
for article in squad_data:
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            if not qa["is_impossible"]:
                qa_pairs.append({
                    "question": qa["question"],
                    "answer": qa["answers"][0]["text"],
                    "context": para["context"] 
                })

# 初始化两个本地模型
model1 = SentenceTransformer('/home/cw/llms/embedding_model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # 模型1
model2 = SentenceTransformer('/home/cw/llms/embedding_model/sungw111/text2vec-base-chinese-sentence')   # 模型2

# 编码所有上下文（作为向量库）
contexts = [item["context"] for item in qa_pairs]
context_embeddings1 = model1.encode(contexts)  # 模型1的向量库
context_embeddings2 = model2.encode(contexts)  # 模型2的向量库

# 评估函数
def evaluate(model, query_embeddings, context_embeddings):
    correct = 0
    for idx, qa in enumerate(qa_pairs[:100]):  # 测试前100条
        # 查找最相似上下文
        sim_scores = util.cos_sim(query_embeddings[idx], context_embeddings)
        best_match_idx = np.argmax(sim_scores)
        # 检查答案是否在匹配段落中
        if qa["answer"] in contexts[best_match_idx]:
            correct += 1
    return correct / len(qa_pairs[:100])

# 编码所有问题
query_embeddings1 = model1.encode([qa["question"] for qa in qa_pairs[:100]])
query_embeddings2 = model2.encode([qa["question"] for qa in qa_pairs[:100]])

# 执行评估
acc1 = evaluate(model1, query_embeddings1, context_embeddings1)
acc2 = evaluate(model2, query_embeddings2, context_embeddings2)

print(f"模型1准确率: {acc1:.2%}")
print(f"模型2准确率: {acc2:.2%}")