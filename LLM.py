import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Qwen1.5-0.5B from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load once (make sure this is cached!)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_length=512, do_sample=True, top_p=0.9, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
