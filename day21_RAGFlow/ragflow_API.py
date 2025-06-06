import requests
#这里的和http://localhost 和RAGFlow的API界面中“API 服务器”的地址保持一致
BASE_URL = 'http://localhost:80/api/v1/'  # 关键修改：确保路径层级正确
API_KEY = 'ragflow-BlYTIzMjhlNDJlMTExZjA4OWQ0OTY4MT'#RAGFlow的API界面中申请的API Key
CHAT_ID = "93fc19ba422611f085d3eea7f2bb8f62"  # 替换为服务端提供的固定 chat_id


def get_answer(user_message):
    """调用 OpenAI 兼容的对话接口"""
    url = f"{BASE_URL}chats_openai/{CHAT_ID}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "ragflow-model",  # 任意值，但字段必须存在
        "messages": [{"role": "user", "content": user_message}],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # 解析 OpenAI 兼容的响应格式
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"]
            return 200, answer, None
        else:
            error_msg = result.get("message", "响应格式不符合预期")
            return response.status_code, None, error_msg

    except requests.exceptions.HTTPError as http_err:
        return response.status_code, None, f"HTTP错误: {http_err}"
    except Exception as err:
        return None, None, f"未知错误: {err}"


def chat():
    """主聊天循环"""
    print("Chatbot initialized. Type 'exit' to end the conversation.")
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Ending the conversation.")
            break

        status_code, answer, error = get_answer(user_message)

        if status_code == 200 and answer:
            print(f"Chatbot: {answer}")
        elif error:
            print(f"Error: {error}")
        else:
            print("Failed to get response")


if __name__ == "__main__":
    chat()