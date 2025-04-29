# benchmark_generate.py
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from generate import _load_model, get_tokenizer, encode_tokens, generate, device_sync

# checkpoint_path = Path("/home/cs601-asures13/kv-quantization/checkpoints/Mistral-7B/model.pth")
checkpoint_path = Path("/home/cs601-mnair12/kv-quantization/checkpoints/Meta-Llama-3.1-8B/model.pth")

tokenizer_path = checkpoint_path.parent / "tokenizer.model"
device = "cuda"
# model_name = "Mistral"
model_name = "Llama3"

precision = torch.bfloat16

print("Loading model...")
model = _load_model(checkpoint_path, device=device, precision=precision, use_tp=False)
model.eval()
tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)


prompt_text = """Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines..."""
encoded_prompt = encode_tokens(tokenizer, prompt_text, bos=True, device=device)

batch_size = 1
draft_model = None
speculate_k = 5
temperature = 0.8
top_k = 200

# ====== 不同max_new_tokens列表 ======
max_new_tokens_list = [128, 256, 512, 1024]

memory_results = []
tokens_sec_results = []

# ====== 正式测试循环 ======
for max_new_tokens in max_new_tokens_list:
    print(f"\n=== Testing max_new_tokens={max_new_tokens} ===")
    
    torch.cuda.reset_peak_memory_stats()
    device_sync(device)

    t0 = time.perf_counter()

    y, _ = generate(
        model,
        encoded_prompt,
        max_new_tokens=max_new_tokens,
    )

    device_sync(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tokens_generated = y.size(-1) - encoded_prompt.size(-1)
    tokens_per_sec = tokens_generated / elapsed
    max_memory_MB = torch.cuda.max_memory_allocated() / 1e9

    print(f"Time: {elapsed:.2f}s, Tokens/sec: {tokens_per_sec:.2f}, Max Memory: {max_memory_MB:.2f} GB")

    memory_results.append(max_memory_MB)
    tokens_sec_results.append(tokens_per_sec)
print("prompt token:",encoded_prompt.size(-1))
# ====== 画图 ======
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(max_new_tokens_list, memory_results, marker='o')
plt.xlabel("Max Output Length (tokens)")
plt.ylabel("Max Memory Usage (GB)")
plt.title("Memory vs Max Output Length")
plt.grid()

plt.subplot(1,2,2)
plt.plot(max_new_tokens_list, tokens_sec_results, marker='o')
plt.xlabel("Max Output Length (tokens)")
plt.ylabel("Tokens per Second")
plt.title("Tokens/sec vs Max Output Length")
plt.grid()

plt.suptitle(f"{model_name} - Inference Benchmark (Memory and Throughput)", fontsize=16)
plt.tight_layout()
plt.savefig(f"{model_name}_benchmark_memory_tokens_sec.png")
plt.show()
