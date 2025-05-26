#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',cache_dir='/root/AI-WSL/models/embedding_model')