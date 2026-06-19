from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://127.0.0.1:30000/v1",
)

model = "Qwen/qwen2.5-coder-14b"

lora = 

messages = [{
    "role": "user",
    "content": "you should fix a vulner"
}]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "chat_template_kwargs": {"enable_thinking":True},
        "separate_reasoning": True
    }
)

