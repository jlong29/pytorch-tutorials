import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# MODEL_NAME = "meta-llama/Llama-2-13b-hf"
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
NTOKENS     = 100

# Enable 8-bit quantization (consider 4-bit for more memory savings)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print(f"Loading {MODEL_NAME} with optimizations......")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with quantization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

# Enable FP16 precision for speed (NOT compatible with 8 bit quantization)
# model = model.half()

# Enable xFormers for Flash Attention (if installed)
try:
    import xformers
    model.gradient_checkpointing_enable()
    model.config.use_cache = True
    print("Using xFormers for optimized attention.")
except ImportError:
    print("xFormers not installed. Using default attention.")

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for mixed precision acceleration
torch.set_num_threads(8)  # Optimize CPU-GPU parallelism

# Define prompt
prompt = "Explain quantum mechanics in simple terms."

# Tokenize input and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating response...")

with torch.inference_mode(), torch.autocast("cuda"):
    output_tokens = model.generate(
        **inputs,
        max_length=NTOKENS,
        use_cache=True,  # Enable KV cache
        num_beams=4,  # Use beam search for faster generation
        top_k=50,  # Random sampling for diversity
        top_p=0.95,  # Nucleus sampling for balance
        temperature=0.7  # Slight randomness
    )

# Decode and print generated text
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("\nGenerated Output:\n", generated_text)
