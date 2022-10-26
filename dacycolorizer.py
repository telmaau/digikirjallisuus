# pos/neg words
# boosters
# caps
# negations
# idioms
# emoticons
# exclamation questionmarks amplyfication

# pos, neg, neu

from typing import Iterable
from spacy.tokens import Token
from digikirjallisuus.sentidav2.constants import read_lexicon
LEXICON=read_lexicon(dictionary="vader")

from spacy import displacy

import argparse
import json
from pathlib import Path
# import spacy




def make_colors(n=5, cmap="RdYlGn"):
    from pylab import cm, matplotlib

    cmap = cm.get_cmap(cmap, n)  # PiYG

    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        yield matplotlib.colors.rgb2hex(rgba)


def print_colors(HEX: Iterable) -> None:
    from IPython.core.display import HTML, display

    for color in HEX:
        display(HTML(f'<p style="color:{color}">{color}</p>'))




########
########
########

import spacy
from spacy.tokens import Span
from spacy import displacy


from digikirjallisuus.sentidav2.getters import make_span_polarity_getter

Span.set_extension(
    "polarity",
    getter=make_span_polarity_getter(),
)



#text = "jeg er ikke længere sur"
#nlp = spacy.load("da_core_news_lg")
#doc = nlp(text)
#sent = [sent for sent in doc.sents][0]
#sent._.polarity
#span = sent



TPL_TOK = """
<mark class="entity" style="background: {bg}; 
padding: 0.45em 0.6em; margin: 0 0.25em; 
line-height: 1; border-radius: 0.35em; 
box-decoration-break: clone; 
-webkit-box-decoration-break: clone">
    {text}
</mark>
"""

def dacy_displacy(span: Span, style="polarity"):
    thresholds = [t / 10 for t in range(-40, 41)]
    sentiment_colors = make_colors(n=len(thresholds))
    sentiment_color_dict = {str(t): c for c, t in zip(sentiment_colors, thresholds)}

    def __normalize(val: float) -> str:
        return str(max(min(round(val, 1), 5), -5))

    pol = span._.polarity
    t_pols = list(filter(lambda p: p, pol.polarities))
    
    c_spans = [
        {"start": tp.span.doc[tp.span.start].idx - span.doc[span.start].idx, "end": tp.span.doc[tp.span.end-1].idx + len(tp.span.doc[tp.span.end-1].text), "label": __normalize(tp.polarity)}
        for tp in t_pols
    ]

    ex = [
        {
            "text": span.text,
            "ents": c_spans,
            "title": None,
        }
    ]
    html = displacy.render(
        ex, style="ent", manual=True, options={"colors": sentiment_color_dict, "template": TPL_TOK}, jupyter=True,
    )
    return html


#nlp = spacy.load("da_core_news_lg")
#doc = nlp(text)
#sents = [sent for sent in doc.sents]
#span = doc[0:]
# displacy.render(doc)


#dacy_displacy(doc[0:])

# show colored text
def color_text(text, nlp, name, save=False):
    
    lowertext = text.lower()#" ".join(lowerwords)
    doc = nlp(lowertext)
    #sents = [sent for sent in doc.sents]
    span = doc[0:]
    html = dacy_displacy(span)
    # Write HTML String to file.html
    #with open(str(name+".pdf"), "w") as file:
    #    file.write(html)
    if save == True:
        html = dacy_displacy(span, jupyter=False)
        output_path = Path(str(name+".html"))
        output_path.open("w", encoding="utf-8").write(html)


    return html



