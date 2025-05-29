from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.schema import TextNode

import json
import torch

# 1. 初始化本地模型
def setup_local_models():
    # 设置本地embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name="/root/AI-WSL/models/embedding_model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 设置本地LLM模型
    llm = HuggingFaceLLM(
        model_name="/root/AI-WSL/models/Qwen/Qwen1.5-1.8B-Chat",
        tokenizer_name="/root/AI-WSL/models/Qwen/Qwen1.5-1.8B-Chat",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
        device_map="auto",
        generate_kwargs={"temperature": 0.3, "do_sample": True}  # 修改为do_sample=True避免警告
    )
    
    # 全局设置
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 512

# 2. 加载数据并处理格式
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = []
    for item in data:
        if isinstance(item, dict):
            # 处理DPR格式数据
            if 'query' in item and 'positive_passages' in item:
                text = f"查询: {item['query']}\n相关文档: {item['positive_passages'][0]['text']}"
            # 处理QA对格式
            elif 'question' in item and 'answer' in item:
                text = f"问题: {item['question']}\n答案: {item['answer']}"
            else:
                continue
        elif isinstance(item, str):
            text = item
        else:
            continue
            
        node = TextNode(text=text)
        nodes.append(node)
    
    return nodes

# 3. 初始化本地模型
setup_local_models()

# 4. 加载数据
data_path = "/root/AI-WSL/project/projectLlamaIndex2/example-query_engine/qa_pairs.json"
nodes = load_data(data_path)

# 5. 示例查询
query = "如何预防机器学习模型过拟合？"

# 案例1：向量检索（使用本地embedding模型）
vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=3)
print("向量检索结果：", [node.text[:50] + "..." for node in vector_retriever.retrieve(query)])

# 案例2：关键词检索（不使用bm25模式）
from llama_index.core import KeywordTableIndex
keyword_index = KeywordTableIndex(nodes)
keyword_retriever = keyword_index.as_retriever(similarity_top_k=3)  # 使用默认模式
print("关键词检索结果：", [node.text[:50] + "..." for node in keyword_retriever.retrieve(query)])

# 案例3：查询引擎（使用本地LLM生成回答）

#向量引擎下LLM生成回答
#query_engine = vector_index.as_query_engine()

#关键词引擎下LLM生成回答
query_engine = keyword_index.as_query_engine()

response = query_engine.query(query)
print("LLM生成回答：", response)

#应用重排序
from llama_index.postprocessor.cohere_rerank import CohereRerank
reranker = CohereRerank(api_key="YOUR_KEY", top_n=2)
reranked_results = reranker.postprocess_nodes(response, query_str=query)
print("重排序后结果：", [res.text for res in reranked_results])



