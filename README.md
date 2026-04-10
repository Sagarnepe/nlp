# Question 1 Character-Level RNN Language Model 

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

  # Larger Corpus 
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