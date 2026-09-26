"""Script-independent normalization of business names and addresses.

Indic scripts are transliterated with one table: the Brahmic Unicode blocks
(Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
Malayalam) share the ISCII layout, so a character's offset inside its block
identifies the same letter in every script. The consonant `skeleton` then
removes the vowel and aspiration differences that remain between a
transliterated name and its English spelling ("kansaltensi" and
"consultancy" both become "knsltns").
"""

import re
import unicodedata


BRAHMIC_BLOCKS = range(0x0900, 0x0D80)
INDEPENDENT_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch",
    0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th",
    0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y", 0x4E: "t", 0x71: "w",
}
VOWEL_SIGNS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
    0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au",
}
NASALS = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}
VIRAMA = 0x4D
# Malayalam chillu letters are final consonants without an inherent vowel.
CHILLU = {0x0D54: "m", 0x0D55: "y", 0x0D56: "l", 0x0D7A: "n", 0x0D7B: "n",
          0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}

# Abbreviations expand to the full word so a transliterated "praaivet" and an
# English "Pvt" reduce to the same skeleton.
LEGAL_FORMS = {
    "pvt": "private", "prv": "private", "ltd": "limited", "inc": "incorporated",
    "corp": "corporation", "co": "company", "cos": "company", "intl": "international",
    "mfg": "manufacturing", "svcs": "services", "bros": "brothers",
}
ADDRESS_WORDS = {
    "street": "st", "saint": "st", "str": "st", "road": "rd", "avenue": "ave",
    "av": "ave", "drive": "dr", "boulevard": "blvd", "lane": "ln",
    "court": "ct", "place": "pl", "highway": "hwy", "suite": "ste",
    "apartment": "apt", "near": "nr", "opposite": "opp", "opp": "opp",
    "north": "n", "south": "s", "east": "e", "west": "w", "floor": "fl",
    "building": "bldg", "number": "no", "nagar": "ngr", "circle": "cir",
    "parkway": "pkwy", "terrace": "ter", "square": "sq", "mount": "mt",
    # French street types: each pair is observed in the unlabeled France pool
    # (e.g. r 357k / rue 729k, bd 25k / boulevard 38k, rte 12k / route 18k).
    "r": "rue", "bd": "blvd", "impasse": "imp", "allee": "all", "chemin": "ch",
    "route": "rte", "cours": "crs", "residence": "res",
}
# "No 12" / "N° 12" precede a house number (India 1.3M, France 165k occurrences).
NUMBER_MARKERS = {"n", "no", "num"}
# Legal forms and credentials that do not identify the business. Credentials
# are only dropped from the core name when written dotted ("D.O."), since some
# are also ordinary words ("do").
LEGAL_WORDS = {"private", "limited", "incorporated", "corporation", "company", "llc", "llp",
               "lp", "plc", "pllc", "pc", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
               "gmbh", "the", "and"}
DOTTED_CREDENTIALS = {"md", "do", "od", "dds", "lcsw", "pa", "dc", "ei"}
RE_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:[a-z]{2,6})(?:\.[a-z]{2})?/?$")
RE_NON_WORD = re.compile(r"[^0-9a-z]+")
RE_DOTTED = re.compile(r"(?<![a-z0-9])(?:[a-z]\.){2,}[a-z]?(?![a-z0-9])")
RE_ALIAS = re.compile(r"(?<![a-z0-9])(?:aka|dba|d/b/a|t/a|f/k/a|fka|trading as|doing business as|formerly)"
                      r"(?![a-z0-9])\s*:?")


def transliterate(text):
    """Map Brahmic-script letters to ASCII; other characters pass through."""
    output, pending_consonant = [], False
    for character in text:
        code = ord(character)
        if code in CHILLU:
            if pending_consonant:
                output.append("a")
            output.append(CHILLU[code])
            pending_consonant = False
            continue
        if code not in BRAHMIC_BLOCKS:
            # A word-final inherent vowel is not pronounced (schwa deletion),
            # which is how English loanwords such as "limited" are written.
            output.append(character)
            pending_consonant = False
            continue
        offset = code & 0x7F
        if offset in CONSONANTS:
            if pending_consonant:
                output.append("a")
            output.append(CONSONANTS[offset])
            pending_consonant = True
        elif offset in VOWEL_SIGNS:
            output.append(VOWEL_SIGNS[offset])
            pending_consonant = False
        elif offset == VIRAMA:
            pending_consonant = False
        elif offset in INDEPENDENT_VOWELS:
            if pending_consonant:
                output.append("a")
            output.append(INDEPENDENT_VOWELS[offset])
            pending_consonant = False
        elif offset in NASALS:
            if pending_consonant:
                output.append("a")
            output.append(NASALS[offset])
            pending_consonant = False
        elif 0x66 <= offset <= 0x6F:
            if pending_consonant:
                output.append("a")
            output.append(str(offset - 0x66))
            pending_consonant = False
        # Nukta, stress and punctuation marks carry no letter of their own.
    return "".join(output)


def to_ascii(text):
    if not text:
        return ""
    text = transliterate(unicodedata.normalize("NFC", text))
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def join_dotted(lowered):
    """"l.l.c." -> "llc", "s.a.r.l" -> "sarl": dotted abbreviations become one token."""
    return RE_DOTTED.sub(lambda m: m.group(0).replace(".", ""), lowered)


def _name_tokens(lowered):
    tokens = []
    for raw in lowered.replace("&", " and ").split():
        domain = RE_DOMAIN.match(raw)
        if domain:
            raw = domain.group(1)
        for token in RE_NON_WORD.split(raw):
            if token:
                tokens.append(LEGAL_FORMS.get(token, token))
    return tokens


def normalize_name(text):
    return " ".join(_name_tokens(join_dotted(to_ascii(text))))


def core_name(text):
    """Normalized name without legal forms (and without dotted credentials)."""
    lowered = to_ascii(text)
    dotted = {m.replace(".", "") for m in RE_DOTTED.findall(lowered)}
    return " ".join(t for t in _name_tokens(join_dotted(lowered))
                    if t not in LEGAL_WORDS and not (t in DOTTED_CREDENTIALS and t in dotted))


def name_variants(text):
    """The normalized name, plus both sides of an "X aka Y" / "X d/b/a Y" alias.

    Only split when both sides keep at least one token, so a brand that merely
    starts with "Aka" or "DBA" stays whole.
    """
    lowered = join_dotted(to_ascii(text))
    full = " ".join(_name_tokens(lowered))
    match = RE_ALIAS.search(lowered)
    if match:
        left = " ".join(_name_tokens(lowered[:match.start()]))
        right = " ".join(_name_tokens(lowered[match.end():]))
        if left and right:
            return list(dict.fromkeys((full, left, right)))
    return [full]


def normalize_address(text):
    tokens = []
    raw_tokens = [t for t in RE_NON_WORD.split(to_ascii(text)) if t]
    for i, token in enumerate(raw_tokens):
        if token in NUMBER_MARKERS and i + 1 < len(raw_tokens) and raw_tokens[i + 1].isdigit():
            continue
        token = (token.lstrip("0") or "0") if token.isdigit() else token
        tokens.append(ADDRESS_WORDS.get(token, token))
    return " ".join(tokens)


def script_flags(text):
    """(has non-Latin letters, has accented Latin letters)."""
    nonlatin = accented = False
    for character in text:
        if ord(character) >= 0x80 and character.isalpha():
            if ord(character) > 0x024F:
                nonlatin = True
            else:
                accented = True
    return nonlatin, accented


SKELETON_RULES = [
    ("x", "ks"), ("ph", "f"), ("sh", "s"), ("ch", "c"), ("ck", "k"), ("q", "k"),
    ("th", "t"), ("dh", "d"), ("bh", "b"), ("gh", "g"), ("kh", "k"), ("jh", "j"),
    ("w", "v"), ("z", "j"),
]
RE_SOFT_C = re.compile(r"c(?=[eiy])")
RE_VOWELS = re.compile(r"[aeiouyh]")
RE_REPEATS = re.compile(r"(.)\1+")


def skeleton_token(token):
    if token.isdigit():
        return token
    for source, target in SKELETON_RULES:
        token = token.replace(source, target)
    token = RE_SOFT_C.sub("s", token).replace("c", "k")
    head = "a" if token[:1] in "aeiouy" else ""
    return RE_REPEATS.sub(r"\1", head + RE_VOWELS.sub("", token)) or token[:1]


def skeleton(normalized_name):
    return " ".join(skeleton_token(t) for t in normalized_name.split())
