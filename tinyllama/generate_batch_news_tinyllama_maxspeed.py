# generate_batch_news_tinyllama_maxspeed.py
from pathlib import Path
from datetime import datetime
import time
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

PROMPT = """

Write a full news article in the style of CNN or DailyMail.
The story should sound realistic, factual, and human-written.
Use natural journalistic language with short and medium-length sentences.
Start with a strong lead paragraph summarizing who, what, where, and when.
Then expand with quotes, context, background, and a final paragraph about next steps or reactions.
Include realistic numbers, dates, and locations.
Add 1–3 short quotes attributed to plausible people (officials, witnesses, or experts).
Use neutral tone — no opinions, exaggeration, or bullet points.
Output only the article text (no headline, no lists, no explanation, no “to summarize”).
End cleanly after several paragraphs.

"""

def main_sequential(n=50, out_dir="TinyLlama V5"):
    """Fastest sequential version"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Maximum speed optimization
    llm = ChatOllama(
        model="tinyllama",
        temperature=0.8,        # Higher for variety in synthetic data
        top_p=0.9,
        num_predict=350,        # Very short output
        num_ctx=256,            # Minimal context
        repeat_penalty=1.02,    # Minimal repetition penalty
        num_thread=12           # Max CPU threads
    )
    
    msg_tmpl = ChatPromptTemplate.from_template(PROMPT)

    total_start = time.perf_counter()
    completed = 0
    
    print(f" MAX SPEED TinyLlama - Generating {n} articles...")
    print(f" Output: {out_path.absolute()}")
    print("-" * 50)

    for i in range(1, n + 1):
        batch_start = time.perf_counter()
        
        # Generate article
        messages = msg_tmpl.format_messages()
        res = llm.invoke(messages)
        gen_time = time.perf_counter() - batch_start

        # Save with minimal overhead
        ts = datetime.now().strftime("%H%M%S")
        fname = out_path / f"news_{i:04d}_{ts}.txt"
        
        with open(fname, "w", encoding="utf-8") as f:
            f.write(res.content.strip())

        completed += 1
        print(f"⚡ {completed:4d}/{n} | {gen_time:5.2f}s | {len(res.content):4d} chars")

    total_elapsed = time.perf_counter() - total_start
    avg_time = total_elapsed / n
    
    print("-" * 50)
    print(f" COMPLETE: {n} articles in {total_elapsed:.2f}s")
    print(f" Average: {avg_time:.2f}s per article")
    print(f"  100k estimate: {(avg_time * 100000 / 3600 / 24):.1f} days")
    print(f" Speed factor: {(8.0 / avg_time):.1f}x faster than Mistral 7B")

async def main_async(n=20, out_dir="TinyLlama Async", concurrent=2):
    """Async version for maximum throughput"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(
        model="tinyllama",
        temperature=0.8,
        top_p=0.9,
        num_predict=350,
        num_ctx=256
    )
    
    msg_tmpl = ChatPromptTemplate.from_template(PROMPT)
    
    async def generate_article(i):
        start = time.perf_counter()
        messages = msg_tmpl.format_messages()
        res = await llm.ainvoke(messages)
        elapsed = time.perf_counter() - start
        
        # Quick save
        ts = datetime.now().strftime("%H%M%S")
        fname = out_path / f"async_{i:04d}_{ts}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(res.content.strip())
        
        print(f"⚡ {i:4d} | {elapsed:5.2f}s | {len(res.content):4d} chars")
        return elapsed
    
    print(f" ASYNC MODE: {n} articles with {concurrent} concurrent")
    print("-" * 50)
    
    total_start = time.perf_counter()
    
    # Process in concurrent batches
    all_times = []
    for batch_start in range(0, n, concurrent):
        batch_end = min(batch_start + concurrent, n)
        tasks = [generate_article(i+1) for i in range(batch_start, batch_end)]
        batch_times = await asyncio.gather(*tasks)
        all_times.extend(batch_times)
    
    total_elapsed = time.perf_counter() - total_start
    avg_time = sum(all_times) / len(all_times)
    
    print("-" * 50)
    print(f" ASYNC COMPLETE: {n} articles in {total_elapsed:.2f}s")
    print(f" Average: {avg_time:.2f}s per article")
    print(f"  100k estimate: {(avg_time * 100000 / 3600 / 24):.1f} days")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "async":
        # Run async version: python script.py async
        asyncio.run(main_async(n=50, concurrent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "large":
        # Run large batch: python script.py large
        main_sequential(n=100)
    else:
        # Run normal fast version
        main_sequential(n=10)
