from collections import Counter
import re
import unicodedata

# ---------------------------
# 1) Prepare corpus with end-of-word "_"
# ---------------------------
def normalize_and_tokenize(text: str):
    # Keep Devanagari letters/digits and spaces; turn punctuation into spaces.
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।,;:“”‘’\"'()\[\]!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split(" ") if w]
    return words

def prepare_corpus(words):
    # words is a list like ["काठमाडौंमा", "हल्का", ...]
    return [list(w) + ["_"] for w in words]

# ---------------------------
# 2) Count adjacent symbol pairs across whole corpus
# ---------------------------
def count_pairs(corpus):
    pairs = Counter()
    for tokens in corpus:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

def choose_top_pair(pairs):
    # Deterministic tie-break: max freq, then lexicographically smallest pair
    max_freq = max(pairs.values())
    best = min([p for p, f in pairs.items() if f == max_freq])
    return best, max_freq

# ---------------------------
# 3) Merge the chosen pair everywhere in the corpus
# ---------------------------
def merge_pair(corpus, pair):
    L, R = pair
    merged = []
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
        merged.append(new_tokens)
    return merged

# ---------------------------
# 4) Learn BPE merges, printing top pair + evolving vocab size
# ---------------------------
def learn_bpe(corpus, num_merges=30, verbose=True):
    merges_with_freq = []

    for step in range(1, num_merges + 1):
        pairs = count_pairs(corpus)
        if not pairs:
            break

        (L, R), freq = choose_top_pair(pairs)
        merges_with_freq.append(((L, R), freq))
        corpus = merge_pair(corpus, (L, R))

        vocab = sorted({sym for tokens in corpus for sym in tokens})
        if verbose:
            print(f"Step {step:02d}: top_pair=({L},{R})  freq={freq}  vocab_size={len(vocab)}")

    final_vocab = sorted({sym for tokens in corpus for sym in tokens})
    return merges_with_freq, final_vocab, corpus

# ---------------------------
# 5) Segmenter using merge ranks (greedy by earliest merge)
# ---------------------------
def build_merge_ranks(merges_with_freq):
    merges = [pair for (pair, _freq) in merges_with_freq]
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
        _, i, (L, R) = min(candidates)  # smallest rank = earliest merge
        tokens = tokens[:i] + [L + R] + tokens[i + 2 :]
    return tokens

# ---------------------------
# Demo run for the assignment requirements
# ---------------------------
if __name__ == "__main__":
    paragraph = (
        "आज बिहान काठमाडौंमा हल्का वर्षा भयो। "
        "मानिसहरू छाता बोकेर कामतिर हिँडे, र सडकमा सवारीको भीड बढ्यो। "
        "विश्वविद्यालयका विद्यार्थीहरू पुस्तकालयमा बसेर अनुसन्धान गर्दै थिए। "
        "मैले एउटा दुर्लभ शब्द अनुकरणीयता सुनेँ र यसको अर्थ खोजेँ। "
        "साँझतिर मौसम खुल्यो र हावामा चिसोपन थपियो।"
    )

    words = normalize_and_tokenize(paragraph)
    corpus = prepare_corpus(words)

    print("Paragraph words (tokenized):")
    print(words)
    print("\nTraining BPE (>=30 merges if possible):")
    merges_with_freq, final_vocab, final_corpus = learn_bpe(corpus, num_merges=30, verbose=True)

    # 2) Show five most frequent merges (these are the first five chosen)
    print("\nFive most frequent merges (chosen earliest):")
    for i, ((L, R), freq) in enumerate(merges_with_freq[:5], start=1):
        print(f"{i}. ({L}, {R})  freq={freq}  -> new_token={L+R}")

    # 2) Resulting five longest subword tokens in final vocab
    longest5 = sorted(final_vocab, key=lambda t: (-len(t), t))[:5]
    print("\nFive longest learned subword tokens (from final vocab):")
    for t in longest5:
        print(f"{t}  (len={len(t)})")

    # 3) Segment 5 different words from the paragraph
    #    Rare word: "अनुकरणीयता" (appears once in this paragraph)
    #    Derived/inflected form: "काठमाडौंमा" (stem + postposition "मा")
    targets = ["काठमाडौंमा", "विद्यार्थीहरू", "अनुकरणीयता", "अनुसन्धान", "चिसोपन"]
    merge_ranks = build_merge_ranks(merges_with_freq)

    print("\nSegmentations (tokens include '_' where applicable):")
    for w in targets:
        seg = segment_word_bpe(w, merge_ranks)
        print(f"{w} -> {' '.join(seg)}")
