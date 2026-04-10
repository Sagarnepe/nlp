import torch
import math

torch.manual_seed(42)

def attention(Q, K, V):
    """
    Compute scaled dot-product attention:
    Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V

    Args:
        Q: [batch_size, seq_len_q, d_k]
        K: [batch_size, seq_len_k, d_k]
        V: [batch_size, seq_len_k, d_v]

    Returns:
        output: [batch_size, seq_len_q, d_v]
        attn_weights: [batch_size, seq_len_q, seq_len_k]
        scores_unscaled: raw QK^T
        scores_scaled: scaled QK^T / sqrt(d_k)
    """
    d_k = Q.size(-1)

    scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
    scores_scaled = scores_unscaled / math.sqrt(d_k)

    attn_weights = torch.softmax(scores_scaled, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights, scores_unscaled, scores_scaled


# Test with random Q, K, V
batch_size = 1
seq_len = 4
d_k = 8
d_v = 8

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, attn_weights, scores_unscaled, scores_scaled = attention(Q, K, V)

# Print results
print("Q:\n", Q)
print("\nK:\n", K)
print("\nV:\n", V)

print("\nUnscaled Attention Scores (QK^T):\n", scores_unscaled)
print("\nScaled Attention Scores (QK^T / sqrt(d_k)):\n", scores_scaled)

print("\nAttention Weight Matrix:\n", attn_weights)
print("\nOutput Vectors:\n", output)

# Softmax stability check
softmax_unscaled = torch.softmax(scores_unscaled, dim=-1)
softmax_scaled = torch.softmax(scores_scaled, dim=-1)

print("\nSoftmax Before Scaling:\n", softmax_unscaled)
print("\nSoftmax After Scaling:\n", softmax_scaled)

print("\nRow sums before scaling:\n", softmax_unscaled.sum(dim=-1))
print("\nRow sums after scaling:\n", softmax_scaled.sum(dim=-1))