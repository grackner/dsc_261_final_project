# generate_batch_news.py
from pathlib import Path
from datetime import datetime
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

PROMPT = """
Create a synthetic news article and highlights in the style of CNN/DailyMail dataset with the following structure:

ARTICLE FORMAT:
- News-style reporting with factual tone
- 500-800 words average length
- Include key details in the first third (inverted pyramid style)
- Cover current events, politics, technology, or human interest
- Use proper journalistic language

HIGHLIGHTS FORMAT:
- 2-3 concise bullet points summarizing key facts
- 50-60 words total
- Capture the most important information
- Written as complete sentences

Generate both the full article and highlights. Ensure the content reflects real-world news reporting style but is entirely fictional.
""".strip()

def main(n=5, out_dir="Mistral 7B Data Generations"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(model="mistral", temperature=0.8, top_p=0.9, repeat_penalty=1.1)
    msg_tmpl = ChatPromptTemplate.from_template(PROMPT)

    total_start = time.perf_counter()
    durations = []

    for i in range(1, n + 1):
        start = time.perf_counter()
        messages = msg_tmpl.format_messages()
        res = llm.invoke(messages)
        end = time.perf_counter()

        elapsed = end - start
        durations.append(elapsed)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = out_path / f"article_{i:02d}_{ts}.txt"

        with open(fname, "w", encoding="utf-8") as f:
            f.write("### PROMPT USED ###\n")
            f.write(PROMPT + "\n\n")
            f.write("### MODEL OUTPUT ###\n")
            f.write(res.content.strip() + "\n")
            f.write(f"\n### GENERATION TIME: {elapsed:.2f} seconds ###\n")

        print(f"[{i}/{n}] Wrote: {fname} ({elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - total_start
    avg_time = sum(durations) / len(durations)
    print(f"\n Finished {n} generations in {total_elapsed:.2f}s (avg {avg_time:.2f}s per article)")

if __name__ == "__main__":
    main(n=5)
