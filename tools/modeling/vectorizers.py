import enum
from typing import Callable
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, Binarizer
from sklearn.utils.validation import check_is_fitted
from .._validation import _validate_raw_docs
from scipy.sparse import csr_matrix
from ..typing import CallableOnStr


class TokenizingVectorizer(TransformerMixin, _VectorizerMixin, BaseEstimator):
    """Vectorizer base class with the standard Scikit-Learn pre/post processing.

    This is an easily extensible base class which provides the pre-processing machinery of
    Scikit-Learn's HashingVectorizer, CountVectorizer, and TfidfVectorizer. It also
    includes normalization as a post-processing option.

    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        norm="l2",
    ):
        # associated with pre/post-processing
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        if tokenizer is not None and token_pattern == r"(?u)\b\w\w+\b":
            self.token_pattern = None
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.norm = norm

    def build_post_pipe(self):
        post_pipe = Pipeline([("norm", None)])
        if self.norm is not None:
            post_pipe.set_params(norm=Normalizer(norm=self.norm, copy=False))
        return post_pipe

    def analyze_docs(self, X):
        analyzer = self.build_analyzer()
        return map(analyzer, X)

    def fit(self, X, y=None):
        # Triggers a parameter validation
        _validate_raw_docs(X)
        self._warn_for_unused_params()
        self._validate_params()
        return self

    def transform(self, X):
        _validate_raw_docs(X)
        self._validate_params()
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"X_types": ["string"]}


class Doc2Vectorizer(TokenizingVectorizer):
    """Doc2Vec Vectorizer for Scikit Learn API with the standard preprocessing.

    Largely derived from gensim.sklearn_api.D2VTransformer. This class exists
    because the Gensim transformer lacks the preprocessing functionality of the
    built-in Scikit-Learn vectorizers.

    """

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        norm="l2",
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        docvecs=None,
        docvecs_mapfile=None,
        comment=None,
        trim_rule=None,
        size=100,
        alpha=0.025,
        window=5,
        min_count=5,
        max_vocab_size=None,
        sample=1e-3,
        seed=1,
        workers=3,
        min_alpha=0.0001,
        hs=0,
        negative=5,
        cbow_mean=1,
        hashfxn=hash,
        epochs=30,
        sorted_vocab=1,
        batch_words=10000,
    ):
        # Related to pre/post-processing
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            lowercase=lowercase,
            token_pattern=token_pattern,
            stop_words=stop_words,
            ngram_range=ngram_range,
            norm=norm,
        )

        # Related to gensim.models.Doc2Vec
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # Related to gensim.models.Word2Vec
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        # Parameter validation
        _validate_raw_docs(X)
        self._warn_for_unused_params()
        self._validate_params()

        docs = list(self.analyze_docs(X))
        tagged_docs = [TaggedDocument(words, [i]) for i, words in enumerate(docs)]
        self.gensim_model_ = Doc2Vec(
            documents=tagged_docs,
            dm_mean=self.dm_mean,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs,
            docvecs_mapfile=self.docvecs_mapfile,
            comment=self.comment,
            trim_rule=self.trim_rule,
            vector_size=self.size,
            alpha=self.alpha,
            window=self.window,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            sample=self.sample,
            seed=self.seed,
            workers=self.workers,
            min_alpha=self.min_alpha,
            hs=self.hs,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn,
            epochs=self.epochs,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words,
        )
        return self

    def get_docvecs(self):
        check_is_fitted(self)
        vecs = [
            self.gensim_model_.docvecs[i]
            for i in range(self.gensim_model_.docvecs.max_rawint + 1)
        ]
        post_pipe = self.build_post_pipe()
        return post_pipe.fit_transform(vecs)

    def transform(self, X):
        check_is_fitted(self)
        _validate_raw_docs(X)
        if isinstance(X, (np.ndarray, pd.Series, pd.DataFrame)):
            X = X.squeeze().tolist()
        docs = list(self.analyze_docs(X))
        vecs = [self.gensim_model_.infer_vector(doc) for doc in docs]
        vecs = np.reshape(np.array(vecs), (len(docs), self.gensim_model_.vector_size))
        post_pipe = self.build_post_pipe()
        return post_pipe.fit_transform(vecs)


class VaderVectorizer(BaseEstimator, TransformerMixin):
    """Extracts VADER polarity scores from short documents.

    Parameters
    ----------
    trinarize : bool, optional
        Convert vector elements to ternary sign indicators -1.0, 0.0, and 1.0. By default False.
    compound_only : bool, optional
        Make vectors with only the compound score, by default False.
    category_only : bool, optional
        Make vectors with only the positive, neutral, and negative scores, by default False.
    preprocessor : CallableOnStr, optional
        Callable for preprocessing text before VADER analysis, by default None.
    norm : str, optional
        Normalization to apply, by default "l2".
    sparse : bool, optional
        Output a sparse matrix, by default True.
    """

    def __init__(
        self,
        trinarize=False,
        compound_only=False,
        category_only=False,
        preprocessor: CallableOnStr = None,
        norm="l2",
        sparse=True,
    ):
        self.trinarize = trinarize
        self.compound_only = compound_only
        self.category_only = category_only
        self.preprocessor = preprocessor
        self.norm = norm
        self.sparse = sparse

    def build_post_pipe(self):
        """Construct postprocessing pipeline based on parameters."""
        post_pipe = Pipeline([("sign", None), ("norm", None), ("spar", None)])
        if self.trinarize:
            post_pipe.set_params(sign=FunctionTransformer(np.sign))
        if self.norm is not None:
            post_pipe.set_params(norm=Normalizer(norm=self.norm))
        if self.sparse:
            post_pipe.set_params(spar=FunctionTransformer(csr_matrix))
        return post_pipe

    def _validate_params(self):
        """Validate some parameters."""
        if self.category_only and self.compound_only:
            raise ValueError(
                "Incompatible: `compound_only=True` and `category_only=True`."
            )
        if self.preprocessor is not None:
            if not isinstance(self.preprocessor, Callable):
                raise TypeError(
                    f"Expected `preprocessor` to be callable, got {type(self.preprocessor)}"
                )

    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names_

    def fit(self, X, y=None):
        """Does nothing except validate parameters and save feature names."""
        self._validate_params()
        _validate_raw_docs(X)
        self.feature_names_ = ["neg", "neu", "pos", "comp"]
        if self.compound_only:
            self.feature_names_ = self.feature_names_[-1:]
        elif self.category_only:
            self.feature_names_ = self.feature_names_[:-1]
        return self

    def transform(self, X):
        """Extracts the polarity scores and applies postprocessing."""
        # Input and param validation
        self._validate_params()
        _validate_raw_docs(X)

        # Squeeze pseudo-1d structures into a list
        if isinstance(X, (np.ndarray, pd.Series, pd.DataFrame)):
            X = X.squeeze().tolist()

        # Apply preprocessing
        docs = X
        if self.preprocessor is not None:
            docs = self.preprocessor(docs)

        # Perform VADER analysis
        sia = SentimentIntensityAnalyzer()
        vecs = [pd.Series(sia.polarity_scores(x)) for x in docs]
        vecs = pd.DataFrame(vecs)
        if self.compound_only:
            vecs = vecs.loc[:, ["comp"]]
        if self.category_only:
            vecs = vecs.loc[:, ["neg", "neu", "pos"]]
        self.feature_names_ = vecs.columns.to_list()

        # Apply postprocessing and return
        post_pipe = self.build_post_pipe()
        return post_pipe.fit_transform(vecs.to_numpy())