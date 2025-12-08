import os
import torch
import gc
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

################################
# -------- LOAD MODEL -------- #
################################
model_dir = "opt125m-model-news"

# Verify CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA not available! Running on CPU.")
else:
    torch.cuda.empty_cache()
    gc.collect()
    print("Running on GPU, cache cleared")

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True  # More efficient loading
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Verify model is on GPU
print(f"Model device: {model.device}")

#################################
# --------- LOAD DATA --------- #
#################################
synthetic_data = "Synthetic Data/data_for_model2b_additional_v2.csv"
synthetic = pd.read_csv("Synthetic Data/data_for_model2b_additional (1).csv", on_bad_lines="skip", engine="python", encoding="latin")
# pd.read_csv(synthetic_data, on_bad_lines="skip", engine="python") # on_bad_lines="skip", engine="python"
person = "Gabi"
input_df = synthetic[synthetic["person"] == person].copy()
input_df["generated_text"] = ""

save_dir = "inference_outputs_testing_unique"
os.makedirs(save_dir, exist_ok=True)

#######################################
# --------- DATA GENERATION --------- #
#######################################
MIN_NEW_TOKENS = 600
MAX_NEW_TOKENS = 1100
batch_size = 15

# Set model to evaluation mode
model.eval()

losses = []
# Disable gradient computation for inference
with torch.no_grad():
    num_batches = (len(input_df) + batch_size - 1) // batch_size
    print(f"num batches = {num_batches}")
    for b in tqdm(range(num_batches), desc="Generating"):
        start = b * batch_size
        end = min((b + 1) * batch_size, len(input_df))
        batch = input_df.iloc[start:end]
        
        # Batched tokenization
        real_articles = batch["cleaner_article"].tolist()
        # Encode inputs/labels
        tokenized_articles = tokenizer(
            real_articles,
            return_tensors="pt",
            padding=True,
            padding_side='left',
            truncation=True,
            max_length=2048
        ).to(model.device)

        outputs = model(
                    input_ids=tokenized_articles["input_ids"],
                    attention_mask=tokenized_articles["attention_mask"],
                    labels=tokenized_articles["input_ids"]
        )
        
        # Compare Generated tokens to real dataset
        # Calculate loss
        losses.append(outputs["loss"].item())
        # Optional: Print GPU memory usage
        if torch.cuda.is_available():
            print(f"Batch {b+1}/{num_batches} | GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

# Average losses
print(f"DONE! Average loss: {sum(losses)/len(losses)}")

# Clean up GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory cleared")
