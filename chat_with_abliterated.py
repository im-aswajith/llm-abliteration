import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Llama-3.2-3B-Instruct-advanced-abliterated"  # or your custom output dir

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,       # or float16 if you prefer
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Chat with abliterated model (type 'exit' to quit)")
print("-" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        break

    messages = [{"role": "user", "content": user_input}]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Model: {response}")
