import re
import html


def normalize(a, language=None):
    """Yields 1 normalized form of entity mention `a` if possible"""
    QUOTECHARS = "'\"〞「❜❞＂”‚〝»‟―‹›❛❮’‘〟❯„‛“❝«"
    if a:
        a = a.replace("_", " ")
        a = re.sub(f"[{QUOTECHARS}]", "", a.replace("&nbsp;", " "))
        a = html.unescape(a).lower().strip().split(" (")[0]
        # no numbers or dates
        if not (a.startswith("<") or re.match("^[0-9-/—]+$", a)):
            if a and language:
                a = stem(a, language)
            if a:
                yield a

SNOWBALL_LANG = {
    'ar': 'arabic',
    'eu': 'basque',
    'ca': 'catalan',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'fi': 'finnish',
    'fr': 'french',
    'de': 'german',
    'el': 'greek',
    'hi': 'hindi',
    'hu': 'hungarian',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'lt': 'lithuanian',
    'ne': 'nepali',
    'no': 'norwegian',
    'pt': 'portuguese',
    'ro': 'romanian',
    'ru': 'russian',
    'sr': 'serbian',
    'es': 'spanish',
    'sv': 'swedish',
    'ta': 'tamil',
    'tr': 'turkish',
}


STEMMERS = {}
def stem(text, code):
    global STEMMERS
    
    from icu_tokenizer import Tokenizer

    tokenizer = Tokenizer(lang=code).tokenize
    lang = SNOWBALL_LANG.get(code)
    if lang:
        if code not in STEMMERS:
            import snowballstemmer

            STEMMERS[code] = snowballstemmer.stemmer(lang)
        return ' '.join(STEMMERS[code].stemWords(tokenizer(text)))
    elif code == 'fa':
        if code not in STEMMERS:
            from PersianStemmer import PersianStemmer

            STEMMERS[code] = PersianStemmer()
        return STEMMERS[code].run(text)
    elif code == 'ja':
        if code not in STEMMERS:
            import MeCab

            STEMMERS[code] = MeCab.Tagger("-Owakati")
        return STEMMERS[code].parse(text).strip()
    else:
        return ' '.join(tokenizer(text))