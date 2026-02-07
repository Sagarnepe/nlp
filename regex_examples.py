import re

# 1. U.S. ZIP Codes
# (?:[-\s]\d{4})?: Optional non-capturing group for hyphen/space and 4 digits
zip_pattern = r'\b\d{5}(?:[-\s]\d{4})?\b'

# 2. Words NOT starting with a capital letter
# [^A-Z]: Negated character class (not a capital A-Z)
# [a-z'-]+: One or more lowercase letters, apostrophes, or hyphens
negation_pattern = r'\b[^A-Z\s\W][a-z\'-]*\b'

# 3. Rich Numbers
# [-+]?: Optional sign
# \d{1,3}(?:,\d{3})*|\d+: Digits with commas OR just digits
# (?:\.\d+)?: Optional decimal
# (?:[eE][-+]?\d+)?: Optional scientific notation
number_pattern = r'[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?'

# 4. Email Spelling Variants
# (?i): Case-insensitive flag
# [-\s\u2013]: Hyphen, space, or en-dash (\u2013)
email_pattern = r'(?i)e[-\s\u2013]?mail'

# 5. Interjection "gooo!"
# \bgo+: 'g' followed by one or more 'o's
# [!.?,\b]?: Optional single trailing punctuation mark or boundary
go_pattern = r'\bgo+([!.?,])?\b'

# 6. Anchors (Question mark + closing quotes/brackets at line end)
# \? : Literal question mark
# [)"\'\]]*: Any number of closing chars
# \s*: Optional trailing spaces
# $: End of line/string
anchor_pattern = r'\?[)"\'\]]*\s*$'


test_text = """
ZIPs: 12345, 12345-6789, 12345 6789.
Words: Apple, don't, state-of-the-art, banana.
Numbers: -1,234.56, +42, 1.23e-4, 5000.
Emails: Email, e-mail, E mail, e–mail.
Go: go, gooo, gooo! goooo?
Lines:
Did he say "yes"?
Is this it?)] 
"""

print("1. ZIP Codes:", re.findall(zip_pattern, test_text))
print("2. Non-Capital Words:", [m for m in re.findall(r'\b\w[\w\'-]*\b', test_text) if not m[0].isupper()])
print("3. Rich Numbers:", re.findall(number_pattern, test_text))
print("4. Email Variants:", re.findall(email_pattern, test_text))
print("5. Gooo Interjections:", [m.group() for m in re.finditer(go_pattern, test_text)])
print("6. Question line ends:", re.findall(anchor_pattern, test_text, re.MULTILINE))