import re
import string
from collections import defaultdict
from functools import lru_cache, singledispatch
from types import MappingProxyType
from typing import Collection, FrozenSet, Set, Type, Union

import gensim.parsing.preprocessing as gensim_pp
import nltk
from nltk.corpus.reader import wordnet
from nltk.sentiment.util import mark_negation as nltk_mark_neg
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from numpy.lib.arraysetops import isin
from sacremoses import MosesDetokenizer

from ..._validation import _invalid_value, _validate_tokens
from ...typing import TaggedTokenSeq, TaggedTokenTuple, TokenSeq, TokenTuple
from ..settings import CACHE_SIZE, DEFAULT_SEP

tb_detokenizer = TreebankWordDetokenizer()

RE_NEG = re.compile(r"_NEG$")

UNIV_TO_WORDNET = MappingProxyType(
    {
        "ADJ": wordnet.ADJ,
        "NOUN": wordnet.NOUN,
        "PRON": wordnet.NOUN,
        "ADV": wordnet.ADV,
        "VERB": wordnet.VERB,
    }
)
"""Mapping of Universal POS tags to Wordnet POS tags."""

PTB_TO_UNIV = MappingProxyType(nltk.tagset_mapping("en-ptb", "universal"))
"""Mapping of Penn Treebank POS tags to Universal POS tags."""

PTB_TO_WORDNET = MappingProxyType(
    {
        "JJ": wordnet.ADJ,
        "JJR": wordnet.ADJ,
        "JJRJR": wordnet.ADJ,
        "JJS": wordnet.ADJ,
        "JJ|RB": wordnet.ADJ,
        "JJ|VBG": wordnet.ADJ,
        "MD": wordnet.VERB,
        "NN": wordnet.NOUN,
        "NNP": wordnet.NOUN,
        "NNPS": wordnet.NOUN,
        "NNS": wordnet.NOUN,
        "NN|NNS": wordnet.NOUN,
        "NN|SYM": wordnet.NOUN,
        "NN|VBG": wordnet.NOUN,
        "NP": wordnet.NOUN,
        "PRP": wordnet.NOUN,
        "PRP$": wordnet.NOUN,
        "PRP|VBP": wordnet.NOUN,
        "RB": wordnet.ADV,
        "RBR": wordnet.ADV,
        "RBS": wordnet.ADV,
        "RB|RP": wordnet.ADV,
        "RB|VBG": wordnet.ADV,
        "VB": wordnet.VERB,
        "VBD": wordnet.VERB,
        "VBD|VBN": wordnet.VERB,
        "VBG": wordnet.VERB,
        "VBG|NN": wordnet.VERB,
        "VBN": wordnet.VERB,
        "VBP": wordnet.VERB,
        "VBP|TO": wordnet.VERB,
        "VBZ": wordnet.VERB,
        "VP": wordnet.VERB,
        "WP": wordnet.NOUN,
        "WP$": wordnet.NOUN,
        "WRB": wordnet.ADV,
    }
)
"""Mapping of Penn Treebank POS tags to Wordnet POS tags."""


def moses_detokenize(tokens: TokenSeq, lang="en"):
    _validate_tokens(tokens, check_str=True)
    detokenizer = MosesDetokenizer(lang=lang)
    return detokenizer.detokenize(tokens)


@singledispatch
def mark_negation(
    tokens: TokenSeq,
    double_neg_flip: bool = False,
    split=False,
    sep: str = DEFAULT_SEP,
) -> TokenSeq:
    """Mark tokens '_NEG' which fall between a negating word and punctuation mark.

    Wrapper for nltk.sentiment.util.mark_negation. Keeps cache
    to reuse previously computed results.

    Parameters
    ----------
    tokens : sequence of str
        Sequence of tokens to mark negated words in.
    double_neg_flip : bool, optional
        Ignore double negation. False by default.
    split: bool, optional
        Break off 'NEG' tags into separate tokens. False by default.
    sep : str, optional
        Separator for 'NEG' suffix.

    Returns
    -------
    sequence of str
        Tokens with negation marked.
    """
    # Fallback dispatch to catch any seq of tokens
    _validate_tokens(tokens)

    # Make immutable (hashable)
    # Send to tuple dispatch for caching
    tokens = mark_negation(
        tuple(tokens),
        double_neg_flip=double_neg_flip,
        split=split,
        sep=sep,
    )

    # Make mutable and return
    return list(tokens)


@mark_negation.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(
    tokens: tuple,
    double_neg_flip: bool = False,
    split=False,
    sep: str = DEFAULT_SEP,
) -> TokenTuple:
    """Dispatch for tuple. Keeps cache to reuse previous results."""
    _validate_tokens(tokens, check_str=True)
    # Make mutable
    tokens = list(tokens)

    # Apply nltk.sentiment.util.mark_negation
    tokens = nltk_mark_neg(tokens, double_neg_flip=double_neg_flip)

    if split:
        # Convert tags into independent 'NEG' tokens
        for i, token in enumerate(tokens):
            if RE_NEG.search(token):
                tokens[i] = token[: token.rfind("_")]
                tokens.insert(i + 1, "NEG")

    elif sep != "_":
        # Subsitute underscore for `sep`
        for i, word in enumerate(tokens):
            tokens[i] = RE_NEG.sub(f"{sep}NEG", word)

    # Make immutable and return
    return tuple(tokens)


@singledispatch
def pos_tag(
    tokens: TokenSeq,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples=False,
    split_tuples=False,
    replace=False,
    sep: str = DEFAULT_SEP,
) -> Union[TaggedTokenSeq, TokenSeq]:
    """Tag `tokens` with parts of speech.

    Wrapper for `nltk.pos_tag`. Keeps cache to reuse
    previous results.

    Parameters
    ----------
    tokens : sequence of str
        Word tokens to tag with PoS.
    tagset : str, optional
        Name of NLTK tagset to use, defaults to Penn Treebank
        if not specified. Unfortunately, NLTK does not have a
        consistent approach to their tagset names.
    lang : str, optional
        Language of `tokens`, by default "eng".
    fuse_tuples : bool, optional
        Join ('token', 'tag') tuples as 'token_tag' according to `sep`.
        By default False.
    split_tuples : bool, optional
        Break up tuples so that tags mingle with the tokens. Equivalent to
        flattening the sequence. By default False.
    replace : bool, optional
        Replace word tokens with their PoS tags, by default False.
    sep : str, optional
        Separator used if `fuse_tuples` is set.

    Returns
    -------
    sequence of tuple of str, or sequence of str
        Tokens tagged with parts of speech, or related sequence of str.
    """
    # Fallback dispatch to catch any seq of tokens
    _validate_tokens(tokens)

    # Make immutable (hashable)
    # Send to tuple dispatch for caching
    tokens = pos_tag(
        tuple(tokens),
        tagset=tagset,
        lang=lang,
        fuse_tuples=fuse_tuples,
        split_tuples=split_tuples,
        replace=replace,
        sep=sep,
    )

    # Make mutable and return
    return list(tokens)


@pos_tag.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(
    tokens: tuple,
    tagset: str = None,
    lang: str = "eng",
    fuse_tuples=False,
    split_tuples=False,
    replace=False,
    sep=DEFAULT_SEP,
) -> TaggedTokenTuple:
    """Dispatch for tuple. Keeps cache to reuse previous results."""
    # Validate params
    _validate_tokens(tokens, check_str=True)
    if sum([fuse_tuples, split_tuples, replace]) > 1:
        raise ValueError(
            "Only one of `fuse_tuples`, `split_tuples`, or `replace` may be True."
        )

    # Tag PoS
    tokens = nltk.pos_tag(tokens, tagset=tagset, lang=lang)

    if fuse_tuples:
        # Fuse tuples
        tokens = [nltk.tuple2str(x, sep) for x in tokens]
    elif split_tuples:
        # Split each tuple into two word tokens
        tokens = [x for tup in tokens for x in tup]
    elif replace:
        # Replace word tokens with PoS tags
        tokens = [t for _, t in tokens]
    return tuple(tokens)


@singledispatch
def wordnet_lemmatize(tokens: TokenSeq) -> TokenSeq:
    """Reduce English words to root form using Wordnet.

    Tokens are first tagged with parts of speech and then
    lemmatized accordingly. Keeps cache to reuse previous
    results.

    Parameters
    ----------
    tokens : sequence of str
        Tokens to lemmatize.

    Returns
    -------
    Sequence of str
        Lemmatized tokens.
    """
    # Fallback dispatch to catch any seq of tokens
    _validate_tokens(tokens)

    # Make immutable (hashable)
    # Send to tuple dispatch for caching
    tokens = wordnet_lemmatize(tuple(tokens))

    # Make mutable and return
    return list(tokens)


@wordnet_lemmatize.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(tokens: tuple) -> TokenTuple:
    """Dispatch for tuple. Keeps cache to reuse previous results."""
    _validate_tokens(tokens, check_str=True)

    # Tag POS using the original (non-caching) function
    tag_toks = nltk.pos_tag(tokens)

    # Convert Penn Treebank tags to Wordnet tags
    ptb2wordnet = defaultdict(lambda: wordnet.NOUN, **PTB_TO_WORDNET)
    tag_toks = [(w, ptb2wordnet[t]) for w, t in tag_toks]

    # Lemmatize
    wnl = WordNetLemmatizer()
    tokens = [wnl.lemmatize(w, t) for w, t in tag_toks]

    # Make immutable and return
    return tuple(tokens)


@singledispatch
def porter_stem(tokens: TokenSeq, lowercase: bool = False) -> TokenSeq:
    """Reduce English words to stems using Porter algorithm.

    Keeps cache to reuse previous results.

    Parameters
    ----------
    tokens : sequence of str
        Tokens to stem.
    lowercase : bool, optional
        Make lowercase, by default False.

    Returns
    -------
    Sequence of str
        Stemmed tokens.
    """
    # Fallback dispatch to catch any seq of tokens
    _validate_tokens(tokens)

    # Make immutable (hashable)
    # Send to tuple dispatch for caching
    tokens = porter_stem(tuple(tokens), lowercase=lowercase)

    # Make mutable and return
    return list(tokens)


@porter_stem.register
@lru_cache(maxsize=CACHE_SIZE, typed=False)
def _(tokens: tuple, lowercase: bool = False) -> TokenTuple:
    """Dispatch for tuple. Keeps cache to reuse previous results."""
    _validate_tokens(tokens, check_str=True)

    # Stem and return
    stemmer = nltk.PorterStemmer()
    return tuple(stemmer.stem(x, lowercase) for x in tokens)


def filter_length(tokens: TokenSeq, min_char=3, max_char=15) -> TokenSeq:
    """Remove tokens with too few or too many characters.

    Parameters
    ----------
    tokens : sequence of str
        Tokens to filter by length.
    min_char : int, optional
        Minimum length, by default 3.
    max_char : int, optional
        Maximum length, by default 15.

    Returns
    -------
    Sequence of str
        Filtered tokens.
    """
    _validate_tokens(tokens, check_str=True)
    if min_char is not None:
        tokens = [x for x in tokens if min_char <= len(x)]
    if max_char is not None:
        tokens = [x for x in tokens if len(x) <= max_char]
    return tokens


def filter_stopwords(
    tokens: TokenSeq, stopwords: Union[str, Set[str]] = "nltk_english"
) -> TokenSeq:
    """Remove stopwords from `tokens`.

    Parameters
    ----------
    docs : sequence of str
        Tokens to remove stopwords from.
    stopwords : str or collection of str, optional
        Set of stopwords, name of recognized stopwords set, or query.
        Defaults to 'nltk_english'.

    Returns
    -------
    Sequence of str
        Tokens with stopwords removed.
    """
    _validate_tokens(tokens, check_str=True)
    if isinstance(stopwords, str):
        stopwords = fetch_stopwords(stopwords)
    else:
        stopwords = set(stopwords)
    return [x for x in tokens if x not in stopwords]


def fetch_stopwords(query: str) -> FrozenSet[str]:
    """Fetch a recognized stopwords set.

    Recognized sets include 'skl_english', 'nltk_english', 'nltk_spanish',
    'nltk_french', 'gensim_english'. Will recognize 'nltk_{language}' in general
    if provided the language (fileid) of an NLTK stopwords set. Supports complex
    queries involving set operators '|', '&', '-', and '^' and parentheses.

    Parameters
    ----------
    query: str
        Name of recognized stopwords set or complex query involving names.

    Returns
    -------
    frozenset of str
        A set of stop words.
    """
    # Validate string
    if not isinstance(query, str):
        raise TypeError(f"Expected `name` to be str, got {type(query)}.")
    # Process string input
    else:
        # Perform complex fetch with set ops
        if set("|&-^") & set(query):
            # Construct Python expression to fetch each set and perform set ops
            expr = re.sub("\w+", lambda x: f"fetch_stopwords('{x[0]}')", query)
            # Restrict symbols
            symbols = set(re.findall(fr"[{string.punctuation}]|\sif\s|\selse\s", expr))
            if not symbols.issubset(set("|&-^_()'")):
                raise ValueError(f"Invalid query: {query}")
            # Evaluate expression
            result = eval(expr)
        # Fetch SKL stopwords
        elif query in {"skl_english", "skl"}:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            result = frozenset(ENGLISH_STOP_WORDS)
        # Fetch NLTK stopwords
        elif query.startswith("nltk"):
            if "_" in query:
                # Split name to get language
                components = query.split("_")
                # Only allow one language at a time (for uniform syntax)
                if len(components) > 2:
                    raise ValueError(f"Invalid query: {query}")
                # NLTK stopwords fileid e.g. 'english', 'spanish'
                fileid = components[1]
                result = frozenset(nltk.corpus.stopwords.words(fileids=fileid))
            else:
                # Defaults to 'english' if no languages specified
                result = frozenset(nltk.corpus.stopwords.words("english"))
        # Fetch Gensim stopwords
        elif query in {"gensim_english", "gensim"}:
            from gensim.parsing.preprocessing import STOPWORDS

            result = frozenset(STOPWORDS)
        # Raise ValueError if unrecognized
        else:
            raise ValueError(f"Invalid query: {query}")
    return result