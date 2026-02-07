                                    NAME : SAGAR NEUPANE
                                    EMAIL : SXN14600@UCMO.EDU
                                    700# : 700791460





For Question 2.2 (bpe_normal) the BPE Learner

Q. How subword tokens solved the OOV (out-of-vocabulary) problem.

    Subword tokens solve OOV because any unseen word can still be expressed as a sequence of smaller known units, so the model does not need a single whole-word entry to represent it. In this toy setup, “widest” was not in the training words, but it can still be tokenized into known pieces ending with est_, so it remains representable instead of becoming an unknown token. This reduces the number of truly unseen inputs, especially for productive word formation like suffixes and compounds. 

Q. One example where subwords align with a meaningful morpheme (e.g., er_ as English agent/comparative suffix).

    Subwords often align with meaningful morphemes when the corpus repeatedly contains those patterns. Here er_ emerges from “newer_” and “wider_”, and it corresponds to the English comparative suffix “-er” plus the end marker. Likewise est_ emerges from “lowest_” and acts like the superlative suffix “-est” plus the end marker.

For Question 2.3 (bpe_nepali)

Q. What kinds of subwords were learned (prefixes, suffixes, stems, whole words)?

    Subword tokenization helps with OOV in Nepali because unseen words can still be represented as sequences of smaller known units, instead of collapsing to a single unknown token. In a Nepali paragraph, BPE often learns frequent stems plus common endings like postpositions and case markers (for example, “मा”, “को”, “ले”) and plural or honorific patterns (like “हरू”), along with frequent character sequences inside common words. It can also learn whole words (especially short, high-frequency ones) as single tokens with _ at the end.

Q. Two concrete pros/cons of subword tokenization for your language.

    A concrete advantage is better coverage for productive word formation: inflected or derived forms like “काठमाडौंमा” can be tokenized using pieces that appeared elsewhere. Another advantage is a smaller vocabulary than word-level tokenization, which improves generalization. A drawback is that learned subwords may split in ways that do not match linguistic morphemes, which can reduce interpretability. Another drawback for Devanagari is that character-level starting points can lead to merges that reflect script-specific sequences rather than meaningful semantic units, depending on the training text size and normalization.


Question 5

Take a short paragraph (3–4 sentences) in your language (e.g., from news, a story, or social media).
Do naïve space-based tokenization.
Manually correct the tokens by handling punctuation, suffixes, and clitics.
Submit both versions and highlight differences.

Sentence : आजै बिहान काठमाडौँ उपत्यकामा हल्का वर्षा भयो, तर कार्यालय समयमै खुल्यो। मैले सामाजिक सञ्जालमा देखेको एउटा पोस्टमा  ‘काठमाडौंमा ट्राफिक जाम निकै बढ्यो!’ भनेर लेखिएको थियो। त्यसैले विद्यार्थीहरूले पनि छाता बोकेर क्याम्पसतिर हिँडे। साँझतिर मौसम खुल्दै गयो र हावामा चिसोपन थपियो।

Naive space-based tokens (split only on spaces)

आजै

बिहान

काठमाडौँ

उपत्यकामा

हल्का

वर्षा

भयो,

तर

कार्यालय

समयमै

खुल्यो।

मैले

सामाजिक

सञ्जालमा

देखेको

एउटा

पोस्टमा

‘काठमाडौंमा

ट्राफिक

जाम

निकै

बढ्यो!’

भनेर

लेखिएको

थियो।

त्यसैले

विद्यार्थीहरूले

पनि

छाता

बोकेर

क्याम्पसतिर

हिँडे।

साँझतिर

मौसम

खुल्दै

गयो

र

हावामा

चिसोपन

थपियो।

Manually corrected tokens (punctuation split; suffixes/clitics handled)

[आज, ै, बिहान, काठमाडौँ, उपत्यका, मा, हल्का, वर्षा, भयो, ,, तर, कार्यालय, समय, मै, खुल्यो, ।, म, ले, सामाजिक, सञ्जाल, मा, देखेको, एउटा, पोस्ट, मा, ‘, काठमाडौं, मा, ट्राफिक, जाम, निकै, बढ्यो, !, ’, भनेर, लेखिएको, थियो, ।, त्यसैले, विद्यार्थी, हरू, ले, पनि, छाता, बोकेर, क्याम्पस, तिर, हिँडे, ।, साँझ, तिर, मौसम, खुल्दै, गयो, र, हावा, मा, चिसो, पन, थपियो, ।]

Highlighted differences (examples)

भयो, → भयो + ,

खुल्यो। → खुल्यो + ।

उपत्यकामा → उपत्यका + मा

समयमै → समय + मै (clitic/emphasis)

सञ्जालमा / पोस्टमा / हावामा → सञ्जाल/पोस्ट/हावा + मा

‘काठमाडौंमा → ‘ + काठमाडौं + मा

बढ्यो!’ → बढ्यो + ! + ’

विद्यार्थीहरूले → विद्यार्थी + हरू + ले

क्याम्पसतिर / साँझतिर → क्याम्पस/साँझ + तिर

मैले → म + ले

चिसोपन → चिसो + पन (derivational “-ness”)

Multiword expressions (MWEs) to treat as single tokens

काठमाडौँ उपत्यका -- A fixed place name; keeping it together helps named-entity recognition and prevents losing the location identity.

सामाजिक सञ्जाल -- A conventional compound meaning “social media”; splitting can blur meaning and harms search/intent matching.

ट्राफिक जाम -- A fixed loan phrase used as one concept; treating it as one token helps sentiment/topic modeling and retrieval.

Q. What was the hardest part of tokenization in your language? How does it compare with tokenization in English?

    The hardest part in Nepali tokenization is that many grammatical markers behave like suffixes or clitics attached to the word (मा, ले, हरू, तिर, पन, ै, मै), so a simple tokenizer misses useful boundaries. Compared with English, Nepali has richer morphology and postpositions that often stick to stems, making “word = token” a weaker assumption. Punctuation like danda (।) and quotation marks also need explicit handling because naïve whitespace tokenization leaves them attached.

Q. Do punctuation, morphology, and MWEs make tokenization more difficult?

    Overall, punctuation, morphology, and MWEs all make Nepali tokenization more difficult than English in different ways: punctuation needs cleaning, morphology needs splitting, and MWEs need merging.