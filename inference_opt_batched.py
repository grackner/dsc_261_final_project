import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

################################
# -------- LOAD MODEL -------- #
################################

model_dir = "opt125m-news-model-base"

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)


#################################
# --------- LOAD DATA --------- #
#################################

synthetic_data = "Synthetic Data/synthetic_data_prepared.csv"
synthetic = pd.read_csv(synthetic_data, on_bad_lines="skip", engine="python")

person = "Kira" # change this 
input_df = synthetic[synthetic["person"] == person].copy()
input_df["generated_text"] = ""

save_dir = "inference_outputs_testing_unique"
os.makedirs(save_dir, exist_ok=True)


#######################################
# --------- DATA GENERATION --------- #
#######################################
MIN_NEW_TOKENS = 600
MAX_NEW_TOKENS = 1100

batch_size = 50
num_batches = (len(input_df) + batch_size - 1) // batch_size

for b in tqdm(range(num_batches), desc="Generating"):
    start = b * batch_size
    end = min((b + 1) * batch_size, len(input_df))
    batch = input_df.iloc[start:end]

    # batched tokenization
    prefixes = batch["prefix"].tolist()
    tokenized_batch = tokenizer(
        prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(model.device)

    # full batch generation
    outputs = model.generate(
        **tokenized_batch,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    # decode outputs per row 
    generated_texts = [
        tokenizer.decode(outputs[i], skip_special_tokens=True)
        for i in range(outputs.shape[0])
    ]

    # store outputs
    input_df.loc[batch.index, "generated_text"] = generated_texts

    # save only current batch 
    partial_path = os.path.join(save_dir, f"generation_batch_{b+1}.csv")
    batch_output = input_df.loc[batch.index, ["id", "generated_text"]]
    batch_output.to_csv(partial_path, index=False)
    print(f"Saved batch {b+1}/{num_batches} to {partial_path}")


##################################
# --------- FINAL SAVE --------- #
##################################

final_path = os.path.join(save_dir, "full_generation_output.csv")
input_df[["id", "generated_text"]].to_csv(final_path, index=False)
print(f"\n🎉 DONE! Full output saved to:\n{final_path}")
