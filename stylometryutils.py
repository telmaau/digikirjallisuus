
# import libraries
import pandas as pd
import numpy as np
import os
#import umap
#import umap.plot
import glob 
import argparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import scipy
import matplotlib.pyplot as plt


import json
import string
from collections import Counter
from utils import mattr, list_sliding_window#, ma_punct
table = str.maketrans(dict.fromkeys(string.punctuation))


# stopwords spacy
from spacy.lang.fi.stop_words import STOP_WORDS
import spacy

# %%

# def get tokens
def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-zåæø]+', text.lower()) 

# def counter
count = lambda l1,l2: sum([1 for x in l1 if x in l2])

# build pipeline
# custom segment featurizer
class SegmentFeaturizer:
    def __init__(self):
        self.name ="english_pipeline"
        self.nlp = spacy.load('fi_core_news_sm')
        #self.stopwords = nlp.Defaults.stop_words

        #self.nlp = spacy.load("da_core_news_sm")
    
    
    @staticmethod
    def get_sent_length_stats(doc):
        token_counts = [len(sent) for sent in doc.sents if len(sent.text)>1]
        punct_counts= [count(sent.text,set(string.punctuation)) for sent in doc.sents if len(sent.text)>1]#/len(doc.sents) # average over the story
        #print(punct_counts)
        sent_length_dict = {
                "segment_length": len(doc),
                "n_sents": len(list(doc.sents)),
                "mean_sent_length": np.mean(token_counts),
                "std_sent_length": np.std(token_counts),
                "mean_punct": np.mean(punct_counts)
            }
        return sent_length_dict

    @staticmethod
    def get_words_stats(doc):
        text = " ".join([token.text for token in doc])
        words = [token.text for token in doc if token.pos_ not in ["SYM","PUNCT", "SPACE"] ] # filter out special characters
        lemmas = [token.lemma_ for token in doc if token.pos_ not in ["SYM","PUNCT", "SPACE"] ]
        token_counts = [len(sent) for sent in doc.sents]
        mymattr = mattr(words, window_size=300)
        ttr = len(set(words))/len(words)
        verbs= [token.text for token in doc if token.pos_ == "VERB" ]
        adjectives = [token.text for token in doc if token.pos_ == "ADJ" ]
        verb_fraction =  len(verbs) / len(words)
        adj_fraction = len(adjectives)/ len(words)
        speech = count(doc.text,set('"'))/len(doc.text)
        sent_length_dict = {
            "story_length": len(doc),
            "words": words,
            "lemmas": lemmas,
            "mattr": mymattr,
            "ttr": ttr,
            "speech": speech,
            "verb_fraction" : verb_fraction,
            "adj_fraction" : adj_fraction,
            "mean_sent_length": np.mean(token_counts),
            "std_sent_length": np.std(token_counts),
        }
        return sent_length_dict
    
    @staticmethod
    def get_n_words_before_main_verb(doc):
        numbers = [0]
        for sent in doc.sents:
            main = [t for t in sent if t.dep_ == "ROOT"][0]
            if main.pos_ == "VERB":
                dist_to_init = main.i - sent[0].i
                numbers.append(dist_to_init)
        return np.mean(numbers)

    @staticmethod
    def get_n_complex_clauses(doc):
        embedded_elements_count = []
        for sent in doc.sents:
            n_embedded = len(
                [t for t in sent if t.dep_ in {"ccomp", "xcomp", "advcl", "dative"}]
            )
            embedded_elements_count.append(n_embedded)
        return np.mean(embedded_elements_count)
    
    # putting it all together!
    def featurize(self, segments):
        feature_dicts = []
        docs = self.nlp.pipe(segments)
        for doc in docs:
            feature_dict = {
                #"words": self.get_words_stats(doc)["words"],
                "mattr": self.get_words_stats(doc)["mattr"],
                "speech": self.get_words_stats(doc)["speech"], 
                #"mattr": self.get_words_stats(doc)["mattr"],
                "verb_fraction": self.get_words_stats(doc)["verb_fraction"],
                #"n_complex_clauses": self.get_n_complex_clauses(doc),
                #"mean_sent_length": self.get_words_stats(doc)["mean_sent_length"],
                "mean_punct": self.get_sent_length_stats(doc)["mean_punct"],
            }
            feature_dicts.append(feature_dict)
        return feature_dicts

    # putting it all together!
    def get_stats(self, segments):
        feature_dict = {}
        docs = self.nlp.pipe(segments)
        for i,doc in enumerate(docs):
            key = str(i)
            feature_dict[key] = self.get_words_stats(doc)
        return feature_dict

# %%


# class object for the custom transformer
class CustomLinguisticFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass    
    def fit(self, x, y=None):
        return self    
    def transform(self, data):
        segment_featurizer = SegmentFeaturizer() 
        return segment_featurizer.featurize(data)



class CustomStatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass    
    def fit(self, x, y=None):
        return self    
    def transform(self, data):
        # set up our featurizer
        segment_featurizer = SegmentFeaturizer() 

        return segment_featurizer.get_stats(data)

