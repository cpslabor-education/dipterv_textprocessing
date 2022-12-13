#szükséges csomagok meghívása
import spacy
import spacy_conll
import huspacy
from spacy import displacy
#nagy magyar nyelvi modell betöltése
nlp = huspacy.load()
#pipeline komponens létrehozása
nlp.add_pipe("conll_formatter", last=True)
#a mondat felbontása tokenekké
doc = nlp("Fazekas Zoltán Magyarországon vásárolta meg jó barátját Bellát a németjuhász kutyát a Kutyafarm Kft.től.")
for token in doc:
    print(token.text)
    # a token több, mint egy String és vannak tulajdonságai.
    # a token típusú osztály olyan, mint egy String, de több attribútuma van

    for token in doc:
        print(
            token.text
            + (14 - len(token.text)) * " "
            + token.text_with_ws
            + (14 - len(token.text_with_ws)) * " "
            + str(token.is_alpha)
            + (7 - len(str(token.is_alpha))) * " "
            + str(token.is_digit)
            + (7 - len(str(token.is_digit))) * " "
            + str(token.is_upper)
            + (7 - len(str(token.is_upper))) * " "
            + str(token.i)
        )
#Szabály alapú lemmatizáló

def print_token_feautres(token):
    print(
        token.text
        + (16 - len(token.text)) * " "
        + token.lemma_
        + (16 - len(token.lemma_)) * " "
        + token.pos_
        + (7 - len(token.pos_)) * " "
        + token.tag_
        + (7 - len(token.tag_)) * " "
        + token.dep_
        + (13 - len(token.dep_)) * " "
        + token.shape_
        + (7 - len(token.shape_)) * " "
        + str(token.is_alpha)
        + (7 - len(str(token.is_alpha))) * " "
        + str(token.is_stop)
    )
# tokenizálás
print("Tokenek:")
print([token.text for token in doc])

# lemmatizálás
print("\n\nLemmák:")
print([token.lemma_ for token in doc])

# POS tagging
print("\n\nPOS tagek:")
for token in doc:
    print_token_feautres(token)

for word in doc:
  print('{} -> {}'.format(word.text, word.morph))

#megjelenítés
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}
sentence_spans = list(doc.sents)
displacy.serve(sentence_spans, style="dep")
displacy.serve(doc, style="dep")
#displacy.serve(doc, style="ent")