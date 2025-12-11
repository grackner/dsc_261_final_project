# DSC 261 Final Project
Directories:
1. Data Eval- Data evaluation directory contains all of the script to measure data quality metrics on synthetic and real data
2. Phi3_Generations- Directory contains the scripts to run inference on Phi model and synthetic datasets
3. Batched_generation folder contains .py file to run batched generation of articles
   - Gemma: run `python batched_text_generation.py` (2.37 hours to generate 12k articles)
   - Phi: run `python batched_text_generation.py --model phi --output phi_outputs.csv`
   - can also specify total articles to generate with  `--n` and batch size with `--batch_size`
4. Qwen folder contains code to run inference on Qwen and some sample outputs
5. synthetic_training_data- contains some of our synthetic training data but most is hosted on Google Drive (see below)
  
Models and Large Files:
1. Models are stored on Google Drive here:
2. Synthetic training data stored here: 
