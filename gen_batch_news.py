# generate_batch_news.py
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

PROMPT = """
You are a professional journalist for a major outlet.
Generate a realistic news ARTICLE and editor-written HIGHLIGHTS in the style of the CNN/DailyMail dataset.
Choose the story angle yourself from common beats (world, US, business, tech, health, environment, sports, science, education, transportation).

Constraints
- Do not mention that this is synthetic or AI-generated.
- Keep names/orgs generic or fictional unless generic institutions (e.g., “the central bank”).
- Neutral newsroom tone; short quotes allowed but avoid specific fabricated stats or unverifiable claims.
- Length: ARTICLE ≈ 600–800 words (≈750 tokens). HIGHLIGHTS total ≤ 60 tokens.
- Structure: dateline on first line (CITY, Mon DD, YYYY —), 5–7 paragraphs, 1–3 sentences each, final paragraph gives background/implications.

Output EXACTLY this format:
ARTICLE:
[full multi-paragraph article]

HIGHLIGHTS:
- point 1
- point 2
- point 3
""".strip()

def main(n=5, out_dir="Mistral 7B Data Generations"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(model="mistral", temperature=0.7, top_p=0.9, repeat_penalty=1.05)
    msg_tmpl = ChatPromptTemplate.from_template(PROMPT)

    for i in range(1, n + 1):
        # build message once per generation
        messages = msg_tmpl.format_messages()
        res = llm.invoke(messages)  # res.content = ARTICLE + HIGHLIGHTS text

        # filename: timestamp + index (avoid overwrites)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = out_path / f"article_{i:02d}_{ts}.txt"

        # write prompt (as header) + a separator + model output
        with open(fname, "w", encoding="utf-8") as f:
            f.write("### PROMPT USED ###\n")
            f.write(PROMPT + "\n\n")
            f.write("### MODEL OUTPUT ###\n")
            f.write(res.content.strip() + "\n")

        print(f"Wrote: {fname}")

if __name__ == "__main__":
    main(n=5)
