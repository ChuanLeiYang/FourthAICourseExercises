import requests
API_URL="https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN="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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
#使用Token访问
response=requests.post(API_URL,headers=headers,json={"inputs":"你好，Hugging face"})
print(response.json())