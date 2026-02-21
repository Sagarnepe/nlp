from collections import Counter, defaultdict
from typing import List, Tuple, Dict

Token = str
Bigram = Tuple[Token, Token]

def read_corpus() -> List[List[Token]]:
    # Training corpus (already tokenized)
    return [
        ["<s>", "I", "love", "NLP", "</s>"],
        ["<s>", "I", "love", "deep", "learning", "</s>"],
        ["<s>", "deep", "learning", "is", "fun", "</s>"],
    ]

def unigram_bigram_counts(corpus: List[List[Token]]) -> Tuple[Counter, Counter]:
    uni = Counter()
    bi = Counter()
    for sent in corpus:
        uni.update(sent)
        bi.update(zip(sent[:-1], sent[1:]))
    return uni, bi

def mle_bigram_prob(bigram: Bigram, uni: Counter, bi: Counter) -> float:
    w1, w2 = bigram
    c_bigram = bi[bigram]
    c_unigram = uni[w1]
    return (c_bigram / c_unigram) if c_unigram > 0 else 0.0

def sentence_probability(sentence: List[Token], uni: Counter, bi: Counter) -> float:
    # P(sentence) = product_i P(w_i | w_{i-1})
    prob = 1.0
    for bg in zip(sentence[:-1], sentence[1:]):
        p = mle_bigram_prob(bg, uni, bi)
        prob *= p
    return prob

def explain_sentence(sentence: List[Token], uni: Counter, bi: Counter) -> None:
    print("Sentence:", " ".join(sentence))
    probs = []
    for bg in zip(sentence[:-1], sentence[1:]):
        p = mle_bigram_prob(bg, uni, bi)
        probs.append((bg, p))
    for (w1, w2), p in probs:
        print(f"  P({w2} | {w1}) = {p:.6f}")
    total = sentence_probability(sentence, uni, bi)
    print(f"  Total sentence probability = {total:.10f}\n")

def main():
    corpus = read_corpus()
    uni, bi = unigram_bigram_counts(corpus)

    print("Unigram counts:")
    for w, c in uni.most_common():
        print(f"  {w:>10} : {c}")

    print("\nBigram counts:")
    for (w1, w2), c in bi.most_common():
        print(f"  ({w1:>5}, {w2:<8}) : {c}")

    s1 = ["<s>", "I", "love", "NLP", "</s>"]
    s2 = ["<s>", "I", "love", "deep", "learning", "</s>"]

    print("\n--- Test sentences ---\n")
    explain_sentence(s1, uni, bi)
    explain_sentence(s2, uni, bi)

    p1 = sentence_probability(s1, uni, bi)
    p2 = sentence_probability(s2, uni, bi)

    if p1 > p2:
        print("Model prefers: <s> I love NLP </s>")
        print("Why: its product of bigram MLE probabilities is higher.")
    elif p2 > p1:
        print("Model prefers: <s> I love deep learning </s>")
        print("Why: its product of bigram MLE probabilities is higher.")
    else:
        print("Model is indifferent: both sentences have equal probability.")

    # Note: With MLE, both sentences should have non-zero probability here
    # because all their bigrams occur in training.

if __name__ == "__main__":
    main()
