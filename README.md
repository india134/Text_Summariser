Text Summarization Using Pretrained T5 Transformer
This repository contains an implementation of text summarization using the pretrained T5 (Text-To-Text Transfer Transformer) model. The model is fine-tuned on a custom dataset to generate concise summaries from longer articles or documents.

Text summarization is the process of creating a shortened version of a document that retains its most important information. This project leverages the T5 Transformer, a state-of-the-art model for natural language processing tasks, for summarizing articles and documents.

Features
Fine-tuning of a pretrained T5 model for abstractive summarization.
Custom data preprocessing and tokenization using Hugging Face's transformers library.
Batch-wise training and evaluation on GPU using PyTorch.
Evaluation metrics: ROUGE-1, ROUGE-2, and ROUGE-L for assessing model performance.
Optimizations for handling large datasets with reduced computational costs.
Dataset
The dataset consists of articles and their corresponding summaries (highlights). The columns include:

article: The input text or document to summarize.
highlights: The ground truth summary for the input text.
Sample Size
For this project:

Training data: Sampled 5000 rows from the original dataset.
Test data: Sampled 1000 rows for evaluation.
Model Architecture
The project uses the T5-small model, which has:

Encoder-Decoder Architecture: The encoder processes the input text, and the decoder generates the summary.
Pretrained weights are fine-tuned on the custom dataset for domain-specific text summarization.
Setup
Prerequisites
Python 3.7+
GPU with CUDA support (optional but recommended for faster training)
Install Dependencies
Clone the repository and install the required packages:

bash
Copy code
git clone https://github.com/yourusername/text-summarization-t5.git
cd text-summarization-t5
pip install -r requirements.txt
Key Libraries
transformers
torch
pandas
scikit-learn
rouge-score
