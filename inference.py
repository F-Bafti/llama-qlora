import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = "./llama-sft-output"

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load base model
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load LoRA adapters on top
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# Test prompt
prompt = """### Instruction:
Write a Python function that reverses a string.

### Input:

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.3
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Fine-tuned Model Output ---")
print(response)

# Save for comparison
with open("finetuned_output.txt", "w") as f:
    f.write(response)
print("\nOutput saved to finetuned_output.txt")