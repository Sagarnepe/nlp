from collections import Counter

# 1) Prepare corpus (character tokens + end-of-word marker "_")
def prepare_corpus(words):
    return [list(w) + ["_"] for w in words]

# 2) Count adjacent pairs across the whole corpus
def count_pairs(corpus):
    pairs = Counter()
    for tokens in corpus:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

# Deterministic tie-break: max freq, then lexicographically smallest pair
def choose_top_pair(pairs):
    max_freq = max(pairs.values())
    best = min([p for p, f in pairs.items() if f == max_freq])
    return best, max_freq

# 3) Merge a pair everywhere in the corpus
def merge_pair(corpus, pair):
    L, R = pair
    merged_corpus = []
    for tokens in corpus:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == L and tokens[i + 1] == R:
                new_tokens.append(L + R)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        merged_corpus.append(new_tokens)
    return merged_corpus

# 4) Learn BPE merges, printing top pair and evolving vocab size
def learn_bpe(corpus, num_merges=50, verbose=True):
    merges = []
    vocab_sizes = []

    for step in range(1, num_merges + 1):
        pairs = count_pairs(corpus)
        if not pairs:
            break

        (L, R), freq = choose_top_pair(pairs)
        merges.append((L, R))
        corpus = merge_pair(corpus, (L, R))

        vocab = sorted({sym for tokens in corpus for sym in tokens})
        vocab_sizes.append(len(vocab))

        if verbose:
            print(f"Step {step:2d}: top pair = ({L}, {R})  freq={freq:2d}  vocab_size={len(vocab)}")

    return merges, vocab_sizes, corpus

# 5) Segment new words using merge ranks (greedy: best-ranked adjacent pair each iteration)
def build_merge_ranks(merges):
    return {pair: rank for rank, pair in enumerate(merges)}

def segment_word_bpe(word, merge_ranks):
    tokens = list(word) + ["_"]
    while True:
        candidates = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_ranks:
                candidates.append((merge_ranks[pair], i, pair))
        if not candidates:
            break

        _, i, (L, R) = min(candidates)  # smallest rank wins
        tokens = tokens[:i] + [L + R] + tokens[i + 2 :]
    return tokens


if __name__ == "__main__":
    text = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    words = text.split()

    corpus = prepare_corpus(words)
    merges, vocab_sizes, final_corpus = learn_bpe(corpus, num_merges=50, verbose=True)

    merge_ranks = build_merge_ranks(merges)

    test_words = ["new", "newer", "lowest", "widest", "newestest"]  # invented word: newestest
    print("\nSegmentations:")
    for w in test_words:
        print(f"{w:>9} -> {' '.join(segment_word_bpe(w, merge_ranks))}")
