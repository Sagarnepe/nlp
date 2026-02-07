import stanza

paragraph = """आजै बिहान काठमाडौँ उपत्यकामा हल्का वर्षा भयो, तर कार्यालय समयमै खुल्यो।
मैले सामाजिक सञ्जालमा देखेको एउटा पोस्टमा ‘काठमाडौंमा ट्राफिक जाम निकै बढ्यो!’ भनेर लेखिएको थियो।
त्यसैले विद्यार्थीहरूले पनि छाता बोकेर क्याम्पसतिर हिँडे।
साँझतिर मौसम खुल्दै गयो र हावामा चिसोपन थपियो।"""

stanza.download("ne")
nlp = stanza.Pipeline("ne", processors="tokenize", tokenize_no_ssplit=False)

doc = nlp(paragraph)
tool_tokens = [t.text for sent in doc.sentences for t in sent.tokens]
print(tool_tokens)