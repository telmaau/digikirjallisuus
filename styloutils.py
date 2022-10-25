import spacy
nlp= spacy.load("fi_core_news_sm")
import sklearn.model_selection as model_selection
import numpy as np
import scipy.spatial.distance as scidist
import sklearn.preprocessing as preprocessing
import sklearn.feature_extraction.text as text

# helper functions
import glob
import re
import advertools as adv

stopwords=adv.stopwords['finnish']
print(sorted(stopwords)[:5])

def clean_text(
    txt: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
    stop_words=stopwords,
    processing="lemmas"
    ) -> str:
    
    """
    A method to clean text 
    """

    if processing == "lemmas":
        doc = nlp(txt)
        #lemmas=[token.lemma_ for token in doc if token.pos_ != "PUNCT"]
        lemmas=[token.lemma_ for token in doc if token.pos_ in ["VERB", "NOUN" ,"ADJ" ,"ADV"]]
        string=" ".join([l for l in lemmas if len(l)>2]) # vain sanat, jotka yli kaksi kirjainta pitkiä

    elif processing =="none":
        # Cleaning the urls
        string = re.sub(r'https?://\S+|www\.\S+', '', txt)

        # Cleaning the html elements
        string = re.sub(r'<.*?>', '', string)

        # Removing the punctuations
        for x in string.lower(): 
            if x in punctuations: 
                string = string.replace(x, "") 

    else:
        # Cleaning the urls
        string = re.sub(r'https?://\S+|www\.\S+', '', txt)

        # Cleaning the html elements
        string = re.sub(r'<.*?>', '', string)

        txt=txt.translate(translate_table)
        stems = stemmer.stem(txt)

        string=" ".join([s for s in stems.split() if len(s)>1])

    # Converting the text to lower
    string = string.lower()

    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()

    return string  

def load_texts(filenames, max_length):
    documents, authors, titles = [], [], []
    storydict={}
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8-sig') as fp:

            # still some punctuation
            fname = filename.replace("'","").replace(" ","-").replace(".txt", "").split("/")[-1]
            astory = fp.readlines()
        author=fname.split("_")[0]
        print(fname)
        #print(storyname)
        astory=" ".join(astory).replace("\n"," ").replace("\'","")
        
        astory= " ".join(astory.split())
        doc = nlp(astory)
        lemmas = [t.lemma_.lower() for t in doc]
        print(lemmas[:5])
        storydict[fname] = {"story": lemmas}
        start_idx, end_idx, segm_cnt = 0, max_length, 1

        # extract slices from the text:
        while end_idx < len(lemmas):
            documents.append(' '.join(lemmas[start_idx:end_idx]))
            authors.append(author)
            titles.append(fname)

            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1
    return documents, authors, titles


def load_text(filename, max_length):
    documents, authors, titles = [], [], []
    
    with open(filename, 'r', encoding='utf-8-sig') as fp:

        # still some punctuation
        fname = filename.replace("'","").replace(" ","-").replace(".txt", "").split("/")[-1]
        astory = fp.readlines()
    author=fname.split("_")[0]
    print(fname)
    #print(storyname)
    astory=" ".join(astory).replace("\n"," ").replace("\'","")
    
    astory= " ".join(astory.split())
    doc = nlp(astory)
    lemmas = [t.lemma_.lower() for t in doc]
    
    start_idx, end_idx, segm_cnt = 0, max_length, 1

    # extract slices from the text:
    while end_idx < len(lemmas):
        documents.append(' '.join(lemmas[start_idx:end_idx]))
        authors.append(author)
        titles.append(fname)

        start_idx += max_length
        end_idx += max_length
        segm_cnt += 1
    return documents, authors, titles



class Delta:
    """Delta-Based Authorship Attributer."""

    def fit(self, X, y):
        """Fit (or train) the attributer.

        Arguments:
            X: a two-dimensional array of size NxV, where N represents
               the number of training documents, and V represents the
               number of features used.
            y: a list (or NumPy array) consisting of the observed author
                for each document in X.

        Returns:
            Delta: A trained (fitted) instance of Delta.

        """
        self.train_y = np.array(y)
        self.scaler = preprocessing.StandardScaler(with_mean=False)
        self.train_X = self.scaler.fit_transform(X)

        return self

    def predict(self, X, metric='cosine'):
        """Predict the authorship for each document in X.

        Arguments:
            X: a two-dimensional (sparse) matrix of size NxV, where N
               represents the number of test documents, and V represents
               the number of features used during the fitting stage of
               the attributer.
            metric (str, optional): the metric used for computing
               distances between documents. Defaults to 'cityblock'.

        Returns:
            ndarray: the predicted author for each document in X.

        """
        X = self.scaler.transform(X)
        dists = scidist.cdist(X, self.train_X, metric=metric)
        return self.train_y[np.argmin(dists, axis=1)]
    
