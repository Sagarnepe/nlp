# Character-Level RNN Language Model 

This project trains a small character-level recurrent neural network to predict the next character from previous characters.

Model:
**Embedding → RNN/GRU/LSTM → Linear → Softmax**

It uses:
- teacher forcing during training
- cross-entropy loss
- Adam optimizer
- temperature-based sampling for text generation

## Files

- **`char_rnn_train.py`**  
  Main script for training, evaluation, plotting, and text generation.

- **`toy_corpus.txt`**  
  Small toy dataset with short repeated words and phrases for quick testing.

- **`alice_excerpt.txt`**  
  Larger plain-text dataset, ideally 50 KB to 200 KB of public-domain text.

- **`runs/`**  
  Output folder created by the program. Example:
  - `runs/toy/`
  - `runs/alice/`

## How to Run

### Toy corpus
```bash
python char_rnn_train.py \
  --text_path toy_corpus.txt \
  --out_dir runs/toy \
  --rnn_type gru \
  --hidden_size 96 \
  --seq_len 40 \
  --batch_size 64 \
  --epochs 8 \
  --steps_per_epoch 60 \
  --start_text "hel"
  ```

  ### Larger Corpus 
  ```bash
  python char_rnn_train.py \
  --text_path alice_excerpt.txt \
  --out_dir runs/alice \
  --rnn_type lstm \
  --hidden_size 128 \
  --seq_len 64 \
  --batch_size 64 \
  --epochs 10 \
  --steps_per_epoch 120 \
  --start_text "Alice "
  ```

# Program Output

## For each run, the program saves:

### loss_curves.png
- Plot of training and validation loss across epochs.

### loss_values.txt
- Numerical loss values in CSV-style format:

### epoch,train_loss,val_loss
- 1,2.845321,2.901224
- 2,2.401992,2.517330

### samples.txt
- Generated text at multiple temperatures such as 0.7, 1.0, and 1.2.



# Mini Transformer Encoder

## Overview
This project implements a **mini Transformer Encoder** in PyTorch to process a small batch of sentences. It demonstrates the core building blocks of the Transformer encoder architecture without using a decoder. The goal is to show how input text is converted into contextual word representations through embedding, positional encoding, self-attention, multi-head attention, feed-forward layers, and normalization.

## Features
- Uses a small dataset of 10 short sentences
- Tokenizes text and builds a vocabulary
- Converts tokens into embeddings
- Adds **sinusoidal positional encoding**
- Implements:
  - Self-attention
  - Multi-head attention
  - Feed-forward network
  - Add & Norm
- Prints:
  - Input tokens
  - Final contextual embeddings
- Displays:
  - Attention heatmap between words

## Files
- `mini_transformer_encoder.py`  
  Main Python script containing the full implementation.


# Scaled Dot-Product Attention

This project implements the attention function used in Transformers:

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

## What it does
- Computes attention scores from **Query (Q)** and **Key (K)**
- Scales scores by **√dₖ**
- Applies **softmax** to get attention weights
- Multiplies attention weights with **Value (V)** to produce output vectors

## Features
- Implemented in **PyTorch**
- Tested with random **Q, K, V** tensors
- Prints:
  - Attention weight matrix
  - Output vectors
  - Softmax values before and after scaling

## Why scaling is important
Without scaling, the dot-product values can become too large, making softmax outputs very peaky and less stable. Dividing by **√dₖ** keeps values in a better range.

## Example output
The program shows:
- Raw attention scores
- Scaled attention scores
- Final attention weights
- Attention output vectors

## Files
- `attention.py` — main implementation and test script