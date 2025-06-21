from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型本地路径
model_path = "/mnt/data/Qwen"  # 请替换成实际路径，如 Qwen1.5-0.5B

# 加载 tokenizer 和模型（使用 CPU）
print("正在加载模型（使用 CPU）...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).float().cpu()
model.eval()

# 添加 .chat 方法（Qwen 模型支持）
def chat(model, tokenizer, query, history=None, max_new_tokens=128):
    if history is None:
        history = []
    # 使用 chat_template 编码
    messages = []
    for past_user_input, past_response in history:
        messages.append({"role": "user", "content": past_user_input})
        messages.append({"role": "assistant", "content": past_response})
    messages.append({"role": "user", "content": query})
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    history.append((query, response))
    return response, history

# 提问内容
question = "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上"

# 聊天（CPU 模式）
response, history = chat(model, tokenizer, question)
print(f"\n你：{question}")
print(f"Qwen：{response}")
