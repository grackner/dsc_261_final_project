import torch
import torch.nn as nn
import uuid
import pandas as pd
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time
import argparse


def load_model_and_tokenizer(model_name):
    """Load tokenizer and model based on command-line argument."""
    if model_name.lower() == "gemma":
        model_id = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    elif model_name.lower() == "phi":
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype='float16')
    #TODO: add line for tinyllama (maybe?)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model


topics = [
    "US - Crime + Justice", "World - Africa", "World - Americas", "World - Asia", "World - Australia",
    "World - China", "World - Europe", "World - India", "World - Middle East", "World - United Kingdom",
    "Politics - CNN Polls", "Politics - Elections", "Business - Tech", "Business - Media", "Business - Markets",
    "Business - Pre-markets", "Business - After-Hours", "Business - Investing", "Business - Markets Now",
    "Health - Fitness", "Health - Food", "Health - Sleep", "Health - Mindfulness", "Health - Relationships",
    "Entertainment - Movies", "Entertainment - Television", "Entertainment - Celebrity", "Tech - Innovate",
    "Tech - Foreseeable Future", "Tech - Innovative Cities", "Style - Arts", "Style - Design", "Style - Fashion",
    "Style - Architecture", "Style - Luxury", "Style - Beauty", "Travel - Destinations", "Travel - Food & Drink",
    "Travel - Lodging and Hotels", "Travel - News", "Sports - Pro Football", "Sports - College Football",
    "Sports - Basketball", "Sports - Baseball", "Sports - Soccer", "Sports - Olympics", "Sports - Hockey",
    "Science - Space", "Science - Life", "Science - Medicine", "Science - Climate", "Science - Solutions",
    "Science - Weather"
]


def generate_prompts(topics, batch_size):
    prompts = []
    chosen_topics = []
    for _ in range(batch_size):
        i = random.randint(0, len(topics) - 1)
        prompt = f"""Write a full news article in the style of CNN or DailyMail.
        The story should sound realistic, factual, and human-written.
        Use natural journalistic language with short and medium-length sentences.
        Start with a strong lead paragraph summarizing who, what, where, and when.
        Then expand with quotes, context, background, and a final paragraph about next steps or reactions.
        Include realistic numbers, dates, and locations.
        Topics can include {topics[i]}.
        Add 1–3 short quotes attributed to plausible people (officials, witnesses, or experts).
        Use neutral tone — no opinions, exaggeration, or bullet points.
        Output only the article text (no headline, no lists, no explanation, no “to summarize”).
        End cleanly after several paragraphs."""
        prompts.append(prompt)
        chosen_topics.append(topics[i])
    return prompts, chosen_topics


def generate_article(tokenizer, model, topics, batch_size):
    prompts, chosen_topics = generate_prompts(topics, batch_size)
    messages = [
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        for prompt in prompts
    ]

    template_messages = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(template_messages, return_tensors="pt", padding=True).to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=750, do_sample=True, temperature=0.9, top_p=0.95, top_k=50
    )

    responses = [
        tokenizer.decode(outputs[i][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        for i in range(batch_size)
    ]
    return responses, chosen_topics


def main():
    parser = argparse.ArgumentParser(description="Generate news articles using Gemma or Phi model.")
    parser.add_argument("--model", type=str, default="gemma", help="Model type: 'gemma' or 'phi'")
    parser.add_argument("--output", type=str, default="gemma_outputs.csv",
                        help="Path to save the generated CSV file")
    parser.add_argument("--n", type=int, default=12000, help="Total number of generated articles")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for generation")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model)

    df = pd.DataFrame(columns=["uuid", "topic", "generated_article"])
    total_time = 0

    for i in range(args.n//args.batch_size):
        df.loc[i, "uuid"] = str(uuid.uuid4())

        start_time = time.perf_counter()
        responses, chosen_topics = generate_article(tokenizer, model, topics, args.batch_size)
        end_time = time.perf_counter()

        #TODO: clean up timing functionality
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        #TODO: clean up process of adding articles to df
        for j in range(args.batch_size):
            df.loc[i * args.batch_size + j, "topic"] = chosen_topics[j]
            df.loc[i * args.batch_size + j, "generated_article"] = responses[j]

        if i % 10 == 0:
            print(i, total_time)
            df.to_csv(args.output, index=False)

    print("Total time:", total_time)


if __name__ == "__main__":
    main()
