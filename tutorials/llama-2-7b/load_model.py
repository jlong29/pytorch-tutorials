import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
NTOKENS    = 200

quant_config = BitsAndBytesConfig(load_in_8bit=True)
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=quant_config,
    device_map="auto"
)

print("Model successfully loaded.")

prompt = "Explain quantum mechanics in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating response...")

with torch.no_grad():
    output_tokens = model.generate(**inputs, max_length=NTOKENS)

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("\nGenerated Output:\n", generated_text)
