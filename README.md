# DSC 261 Final Project
Files:
1. Qwen_text_generation.ipynb- Notebook generating news article synthetic data from Qwen 2B
2. Qwen_outputs.csv- Qwen synthetic data generated from notebook (1)
3. data_cleaning.ipynb- Notebook to clean data before training model. Removes titles and authors
4. gemma_text_generation.ipynb- Notebook generating news article synthetic data from Gemma 2B
   - displays synthetic data generated
5. gen_batch_news.py - Python file running Mistral 7B to generate synthetic data
6. Mistral 7B Data Generations - .txt files with Mistral 7B generations

New:
1. batched_generation folder contains .py file to run batched generation of articles
   - Gemma: run `python batched_text_generation.py` (2.37 hours to generate 12k articles)
   - Phi: run `python batched_text_generation.py --model phi --output phi_outputs.csv`
   - can also specify total articles to generate with  `--n` and batch size with `--batch_size`
