
#从核心包中导入SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader

#单个文件读取示例
reader = SimpleDirectoryReader(
    input_files=["/root/AI-WSL/project/projectLlamaIndex2/example-SimpleDirectoryReader/data/report_with_table.pdf"]
)

# 多文件读取示例，传入文件夹就行
# reader = SimpleDirectoryReader(
#     "/root/AI-WSL/project/projectLlamaIndex2/example-SimpleDirectoryReader/data"
# )
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")#输出加载的文档数量
print(docs)# 输出文档内容

# # 案例2：高级解析
#pip install pdfplumber -i https://pypi.tuna.tsinghua.edu.cn/simple
import pdfplumber

with pdfplumber.open("/root/AI-WSL/project/projectLlamaIndex2/example-SimpleDirectoryReader/data/report_with_table.pdf") as pdf:
    # 提取所有文本
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    print(text[:200])  # 打印前200字符

    # 提取表格（自动检测）
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            print("\n表格内容：")
            for row in table:
                print(row)