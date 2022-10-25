## helper functions
import re
import json

# metadata
# mapping function
def work_vals(fpath, target):
    """ extract all values for work on filepath
    """
    with open(fpath) as fname:
        metadata = json.load(fname)

    return metadata[target]

# tokens
def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-zåæø]+', text.lower()) 

def chunks(xs, n):
    res=[]
    n = max(1, n)
    res= (xs[i:i+n] for i in range(0, len(xs), n))
    reslist= [r for r in res]
    return reslist

# lexical richness measures
from collections import Counter
from itertools import islice




def list_sliding_window(sequence, window_size=2):
    """ Returns a sliding window generator (of size window_size) over a sequence. Taken from
        https://docs.python.org/release/2.3.5/lib/itertools-example.html
        Example
        -------
        List = ['a', 'b', 'c', 'd']
        window_size = 2
        list_sliding_window(List, 2) -> ('a', 'b')
                                        ('b', 'c')
                                        ('c', 'd')
        Parameters
        ----------
        sequence: sequence (string, unicode, list, tuple, etc.)
            Sequence to be iterated over. window_size=1 is just a regular iterator.
        window_size: int
            Size of each window.
        Returns
        -------
        Generator
    """
    iterable = iter(sequence)
    result = tuple(islice(iterable, window_size))
    if len(result) == window_size:
        yield result
    for item in iterable:
        result = result[1:] + (item,)
        yield result

def mattr(words, window_size=100):
    """ Moving average TTR (MATTR) computed using the average of TTRs over successive segments
        of a text.
        Estimate TTR for tokens 1 to n, 2 to n+1, 3 to n+2, and so on until the end
        of the text (where n is window size), then take the average.
        (Covington 2007, Covington and McFall 2010)
        Helper Function
        ---------------
        list_sliding_window(sequence, window_size):
            Returns a sliding window generator (of size window_size) over a sequence
        Parameter
        ---------
        window_size: int
            Size of each sliding window.
        Returns
        -------
        float
    """
    if window_size > len(words):
        raise ValueError('Window size must not be greater than text size of {}. Try a smaller window size.'.format(words))

    if window_size < 1 or type(window_size) is float:
        raise ValueError('Window size must be a positive integer.')

    scores = [len(set(window)) / window_size
        for window in list_sliding_window(words, window_size)]


    mattr = int(sum(scores))/int(len(scores))

    return mattr


import glob
import re
import advertools as adv

stopwords=adv.stopwords['finnish']
print(sorted(stopwords)[:5])

def clean_text(
    txt: str, 
    nlp,
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