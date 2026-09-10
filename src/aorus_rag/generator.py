from llama_cpp import Llama


MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"


print("正在載入模型...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False,
)

print("模型載入完成")


response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "請用繁體中文回答：1+1 等於多少？"
        }
    ],
    max_tokens=100,
    temperature=0.1,
)


answer = response["choices"][0]["message"]["content"]

print()
print("模型回答：")
print(answer)