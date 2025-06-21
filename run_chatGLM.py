from transformers import AutoTokenizer, AutoModel

# 设置本地模型路径
model_path = "/mnt/data/chatglm3-6b"  # 根据实际情况修改为你本地的路径

# 加载本地的 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cpu()  # 使用 CPU 进行推理

# 将模型设置为评估模式
model = model.eval()

# 进行一次对话
question = "我今天很难过，因为我朋友不理我了，你怎么看？"
response, history = model.chat(tokenizer, question, history=[])

# 输出结果
print(f"问题：{question}")
print(f"ChatGLM：{response}")
