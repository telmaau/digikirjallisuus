"""
This contains constants used for Sentida such as lexicon
"""

import os
from inspect import getsourcefile


def read_lexicon(dictionary="vader"):
    if dictionary == "vad":
        lexicon_file = "../../lexicons/vad_scaled.csv"
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), lexicon_file
        )

        with open(lexicon_full_filepath) as f:
            pairs = filter(lambda x: x, f.read().split("\n")[1:])
            lexicon = {word: float(rating) for word, rating in map(lambda x: x.split(","), pairs)}
    elif dictionary =="vader":
        lexicon_file = "lexica/vader_sentiment.txt"
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), lexicon_file
        )

        #with open(lexicon_full_filepath) as f:
        #    pairs = filter(lambda x: x, f.read().split("\n")[:1])
        #    lexicon = {word: float(rating) for word, rating in map(lambda x: x.split("\t"), pairs)}
            
        with open(lexicon_full_filepath) as fp:
            lex=fp.readlines()

        #lex = lex.split("\n")
        
        lines=[]
        lexicon={}
        for l in lex:
            l=l.split("\t")
            lexicon[l[0]] =float(l[1])

    else:
        lexicon_file = "../../lexicons/sentidav2_lemmas.csv"
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(
            os.path.dirname(_this_module_file_path_), lexicon_file
        )
        

        with open(lexicon_full_filepath) as f:
            pairs = filter(lambda x: x, f.read().split("\n")[1:])
            lexicon = {word: float(rating) for word, rating in map(lambda x: x.split(","), pairs)}
    return lexicon


LEXICON = read_lexicon(dictionary="vader")

NEGATIONS = ["ikke", "ik", "ikk", "ik'", "aldrig", "ingen"]

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733  # capitatilization scaler
N_SCALAR = -0.74

BUT_WORDS = {"men", "dog"}
BEFORE_BUT_SCALAR = 0.5
AFTER_BUT_SCALAR = 1.5


INTENSIFIER_DICT = {
    "absolut": B_INCR,
    "utrolig": B_INCR,
    "vældig": B_INCR,
    "helt": B_INCR,
    "betydende": B_INCR,
    "betydelig": B_INCR,
    "bestemt": B_INCR,
    "enorm": B_INCR,
    "exceptionel": B_INCR,
    "ekseptionel": B_INCR,
    "extrem": B_INCR,
    "yderst": B_INCR,
    "fantastisk": B_INCR,
    "flipping": B_INCR,
    "flippin": B_INCR,
    "frackin": B_INCR,
    "fracking": B_INCR,
    "fricking": B_INCR,
    "frickin": B_INCR,
    "frigging": B_INCR,
    "friggin": B_INCR,
    "fuckin": B_INCR,
    "fucking": B_INCR,
    "fuggin": B_INCR,
    "fugging": B_INCR,
    "hella": B_INCR,
    "intensiv": B_INCR,
    "mest": B_INCR,
    "særskilt": B_INCR,
    "ganske": B_INCR,
    "væsentlig": B_INCR,
    "total": B_INCR,
    "uber": B_INCR,
    "virkelig": B_INCR,
    "temmelig": 0.1,
    "megen": 0.2,
    "mega": 0.4,
    "lidt": -0.2,
    "ekstrem": 0.4,
    "total": 0.2,
    "utrolig": 0.3,
    "rimelig": 0.1,
    "seriøs": 0.3,
}
