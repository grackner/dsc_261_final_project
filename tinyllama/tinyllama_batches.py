import uuid
import pandas as pd
import random
import time
import argparse
import requests
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# ========== CONFIG ==========
OLLAMA_URL = "http://localhost:11434/api/generate"
# ============================

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


# def generate_article(topics, batch_size):
#     prompts, chosen_topics = generate_prompts(topics, batch_size)

#     # Use Ollama API for TinyLlama
#     responses = []
#     for prompt in prompts:
#         payload = {
#                 "model": "tinyllama",
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {
#                     "temperature": 0.9,
#                     "top_p": 0.95,
#                     "top_k": 50
#                 },
#                 "num_predict": 750
#             }
#         try:
#                 resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
#                 resp.raise_for_status()
#                 data = resp.json()
#                 responses.append(data.get("response", "").strip())
#         except Exception as e:
#                 responses.append(f"Error generating article: {e}")
#     return responses, chosen_topics

def generate_single_article(prompt):
    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 50
        },
        "num_predict": 750
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"Error generating article: {e}"

def generate_article(topics, batch_size, max_workers=4):
    prompts, chosen_topics = generate_prompts(topics, batch_size)
    
    # Utilize concurrent futures
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        responses = list(executor.map(generate_single_article, prompts))
    
    return responses, chosen_topics


def main():
    parser = argparse.ArgumentParser(description="Generate news articles using TinyLlama (via Ollama).")
    parser.add_argument("--output", type=str, default="outputs.csv", help="Path to save the generated CSV file")
    parser.add_argument("--n", type=int, default=10, help="Total number of generated articles")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    all_results = []
    total_time = 0

    for i in range(args.n // args.batch_size):
        start_time = time.perf_counter()
        responses, chosen_topics = generate_article(topics, args.batch_size, max_workers=args.max_workers)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        total_time += elapsed_time

        for j in range(args.batch_size):
            all_results.append({
                "uuid": str(uuid.uuid4()),
                "topic": chosen_topics[j],
                "generated_article": responses[j]
            })

        if i % 2 == 0:
            print(f"Batch {i}, time so far: {total_time:.2f}s")

            pd.DataFrame(all_results).to_csv(args.output, index=False)

    print("Generation complete. Total time:", total_time)
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
