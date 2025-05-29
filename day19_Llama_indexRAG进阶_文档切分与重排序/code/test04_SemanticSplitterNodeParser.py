from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os


# 2. 加载文档
documents = SimpleDirectoryReader(input_files=["/root/AI-WSL/project/projectLlamaIndex2/example-TextSplit/data/ai.txt"]).load_data()

# # 3. 筛选Markdown文档
# md_docs = [d for d in documents if d.metadata["file_path"].endswith(".md")]

# 4. 初始化模型和解析器
embed_model = HuggingFaceEmbedding(
    #指定了一个预训练的sentence-transformer模型的路径，用于将句子转换为词向量，词向量模型随便选一个就行，
    # 只要能保证文本能被正确编码成向量即可
    model_name="/root/AI-WSL/models/embedding_model/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#设置语义切分器，buffer_size=1表示每次处理一个文档，
# breakpoint_percentile_threshold=90表示使用90%的分割点作为断点
semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=90,
    embed_model=embed_model
)

# 5. 执行语义分割
semantic_nodes = semantic_parser.get_nodes_from_documents(documents)

# 6. 打印结果
print(f"语义分割节点数: {len(semantic_nodes)}")
for i, node in enumerate(semantic_nodes[:2]):  # 只打印前两个节点
    print(f"\n节点{i+1}:\n{node.text}")
    print("-"*50)