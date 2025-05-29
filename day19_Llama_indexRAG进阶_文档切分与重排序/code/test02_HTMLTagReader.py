from llama_index.readers.file import HTMLTagReader

reader = HTMLTagReader(tag="section", ignore_no_id=True)
docs = reader.load_data(
    "https://finance.eastmoney.com/a/202505283416179250.html"
)

for doc in docs:
    print(doc.metadata)

print(docs)