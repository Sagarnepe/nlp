import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1. Small dataset
# -----------------------------
sentences = [
    "i love deep learning",
    "transformers are very powerful",
    "attention helps models focus",
    "pytorch makes building models easy",
    "natural language processing is fun",
    "we study neural networks today",
    "this encoder reads a sentence",
    "self attention connects all words",
    "positional encoding keeps word order",
    "feed forward layers refine features"
]

# -----------------------------
# 2. Tokenization + Vocabulary
# -----------------------------
special_tokens = ["<PAD>", "<UNK>"]

tokenized_sentences = [s.lower().split() for s in sentences]

vocab = {}
for token in special_tokens:
    vocab[token] = len(vocab)

for sent in tokenized_sentences:
    for word in sent:
        if word not in vocab:
            vocab[word] = len(vocab)

id_to_word = {idx: word for word, idx in vocab.items()}

max_len = max(len(s) for s in tokenized_sentences)

def encode_sentence(tokens, vocab, max_len):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

input_ids = torch.tensor(
    [encode_sentence(s, vocab, max_len) for s in tokenized_sentences],
    dtype=torch.long
)

print("Vocabulary size:", len(vocab))
print("Max sentence length:", max_len)
print("\nInput token IDs:")
print(input_ids)

print("\nInput tokens:")
for i, sent in enumerate(tokenized_sentences):
    padded = sent + ["<PAD>"] * (max_len - len(sent))
    print(f"Sentence {i+1}: {padded}")

# -----------------------------
# 3. Sinusoidal Positional Encoding
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# -----------------------------
# 4a. Self-Attention
# -----------------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.query = nn.Linear(d_model, head_dim)
        self.key = nn.Linear(d_model, head_dim)
        self.value = nn.Linear(d_model, head_dim)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        # scores: (batch, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# -----------------------------
# 4b. Multi-Head Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(d_model, self.head_dim) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        head_outputs = []
        head_attn = []

        for head in self.heads:
            out, attn = head(x, mask)
            head_outputs.append(out)
            head_attn.append(attn)

        concat = torch.cat(head_outputs, dim=-1)   # (batch, seq_len, d_model)
        output = self.out_proj(concat)

        # stack attention maps: (batch, num_heads, seq_len, seq_len)
        attn_maps = torch.stack(head_attn, dim=1)
        return output, attn_maps

# -----------------------------
# 4c. Feed-Forward
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 4d. Add & Norm + Encoder Block
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, attn_maps = self.mha(x, mask)
        x = self.norm1(x + attn_output)   # Add & Norm
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)     # Add & Norm
        return x, attn_maps

# -----------------------------
# Full Mini Transformer Encoder
# -----------------------------
class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)     # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x, attn_maps = self.encoder_block(x, mask)
        return x, attn_maps

# -----------------------------
# Padding mask
# -----------------------------
def create_padding_mask(input_ids, pad_idx=0):
    # input_ids: (batch, seq_len)
    # mask: (batch, 1, seq_len) -> later broadcast to (batch, seq_len, seq_len)
    mask = (input_ids != pad_idx).unsqueeze(1)
    return mask

# -----------------------------
# Hyperparameters
# -----------------------------
d_model = 16
num_heads = 2
d_ff = 32

model = MiniTransformerEncoder(
    vocab_size=len(vocab),
    d_model=d_model,
    max_len=max_len,
    num_heads=num_heads,
    d_ff=d_ff
)

# -----------------------------
# Forward pass
# -----------------------------
mask = create_padding_mask(input_ids, pad_idx=vocab["<PAD>"])
contextual_embeddings, attn_maps = model(input_ids, mask)

print("\nFinal contextual embeddings shape:", contextual_embeddings.shape)
# shape: (batch_size, seq_len, d_model)

# -----------------------------
# 5a. Show final contextual embeddings
# -----------------------------
sample_idx = 0
sample_tokens = tokenized_sentences[sample_idx] + ["<PAD>"] * (max_len - len(tokenized_sentences[sample_idx]))

print(f"\nSample sentence tokens:")
print(sample_tokens)

print(f"\nFinal contextual embeddings for Sentence {sample_idx+1}:")
for tok, emb in zip(sample_tokens, contextual_embeddings[sample_idx]):
    print(f"{tok:12s} -> {emb.detach().numpy()}")

# -----------------------------
# 5b. Attention heatmap
# -----------------------------
# Use first sentence, first head
head_idx = 0
attn_matrix = attn_maps[sample_idx, head_idx].detach().numpy()

plt.figure(figsize=(8, 6))
plt.imshow(attn_matrix, aspect="auto")
plt.colorbar()
plt.xticks(range(max_len), sample_tokens, rotation=45, ha="right")
plt.yticks(range(max_len), sample_tokens)
plt.title(f"Attention Heatmap - Sentence {sample_idx+1}, Head {head_idx+1}")
plt.xlabel("Key tokens")
plt.ylabel("Query tokens")
plt.tight_layout()
plt.show()