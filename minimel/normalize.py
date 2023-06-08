import re
import html
import logging


def normalize(a, language=None, spacy=False):
    """Yields 1 normalized form of entity mention `a` if possible"""
    BADCHARS = "'\"〞「❜❞＂”‚〝»‟―‹›❛❮’‘〟❯„‛“❝«()"
    if a:
        a = a.split("_(")[0].replace("_", " ")
        a = re.sub(f"[{BADCHARS}]", "", a.replace("&nbsp;", " "))
        a = html.unescape(a).lower().strip()
        # no numbers or dates
        if not (a.startswith("<") or re.match("^[0-9-/—]+$", a)):
            if a and language:
                a = stem(a, language, spacy=spacy)
            if a:
                yield a


SNOWBALL_LANG = {
    "ar": "arabic",
    "eu": "basque",
    "ca": "catalan",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "ga": "irish",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sr": "serbian",
    "es": "spanish",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
}


STEMMERS = {}
SPACY_MODEL = None


def stem(text, code, spacy=False):
    if spacy:
        global SPACY_MODEL

        import spacy as sp

        if SPACY_MODEL is None:
            SPACY_MODEL = sp.load(code)

        doc = SPACY_MODEL(text)
        # Lemmatization instead of stemming
        return " ".join(token.lemma_ for token in doc)

    else:
        global STEMMERS

        from icu_tokenizer import Tokenizer

        tokenizer = Tokenizer(lang=code).tokenize
        lang = SNOWBALL_LANG.get(code)
        if lang:
            if code not in STEMMERS:
                import snowballstemmer

                STEMMERS[code] = snowballstemmer.stemmer(lang)
            return " ".join(STEMMERS[code].stemWords(tokenizer(text)))
        elif code == "fa":
            if code not in STEMMERS:
                from PersianStemmer import PersianStemmer

                STEMMERS[code] = PersianStemmer()
            return STEMMERS[code].run(text)
        elif code == "ja":
            if code not in STEMMERS:
                import MeCab

                STEMMERS[code] = MeCab.Tagger()
            if not text.strip():
                return ""
            analysis = STEMMERS[code].parse(text).split("\n")[:-2]
            columns = tuple(zip(*[l.split("\t") for l in analysis]))
            try:
                return " ".join(columns[2]).strip()
            except IndexError:
                logging.warn("Bad Japanese: " + text)
                return ""
        else:
            return " ".join(tokenizer(text))
