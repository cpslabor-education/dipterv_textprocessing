#!/usr/bin/env python
# coding: utf-8

#pip install whatlies[spacy]
#pip install umap-learn==0.5.2
#pip install floret


# A medium és large spaCy modellek tartalmaznak szóvektorokat. ^1 Jelenleg még csak néhány nyelven érhető el floret modell, egyelőre a többi nyelven még Word2Vec vektor elérhető.
#
# Egyes esetekben előfordulhat, hogy egy-egy szóhoz nincs társított vektor az előre tanított modellben. Ezeket OOV (out-of-vocabulary), szótáron kívüli szavaknak nevezzük.


import spacy


# magyas és angol modell letöltése
#!python -m spacy download en_core_web_md
#!pip install https://huggingface.co/huspacy/hu_core_news_lg/resolve/main/hu_core_news_lg-any-py3-none-any.w#



#A whatlies csomag adatvizualizációt hoz létre, amiben a szavakat a vektortérben rendezi el.

from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca, Umap, Tsne

lang = SpacyLanguage("en_core_web_md")
words = ["cat", "dog", "fish", "kitten", "man", "woman",
         "king", "queen", "doctor", "nurse"]

emb = EmbeddingSet(*[lang[w] for w in words])
emb.plot_interactive(x_axis=emb["man"], y_axis=emb["woman"])

lang_en = SpacyLanguage("en_core_web_md")

words_en = ["desk", "table", "chair", "banana", "orange", "green", "dog",
            "cat", "fox", "puppy", "monkey", "prince", "king", "queen",
            "princess", "man", "woman", "human", "football", "volleyball",
            "basketball", "handball", "tree", "bush", "flower", "running",
            "marathon", "cycling", "endurance", "ultra"]

emb_en = lang_en[words_en]
pca_en = emb_en.transform(Pca(2)).plot_interactive(title="English: PCA")
tsne_en = emb_en.transform(Tsne(n_components=2, random_state=0, n_iter=10000, perplexity=2)).plot_interactive(title="English: t-SNE")
umap_en = emb_en.transform(Umap(2)).plot_interactive(title="English: UMAP")


# 3 dimenzióredukáló algoritmus outputja:
#
# PCA: Principal Component Analysis
# t-SNE: t-distributed Stochastic Neighbour Embedding
# UMAP: Uniform Manifold Approximation and Projection
# Bővebben: Dimensionality Reduction for Data Visualization: PCA vs TSNE vs UMAP vs LDA


pca_en | tsne_en | umap_en


# Magyar

lang_hu = SpacyLanguage("hu_core_news_lg")


words_hu = ['disznó', 'ló', 'mókus', 'férfi', 'fiú',
         'gyerek', 'korcsolya', 'bicikli', 'hajó', 'kerékpár',
         'lány','nő', 'király', 'kutya', 'macska', 'hercegnő',
         'királylány', 'királynő', 'doktor', 'nővér', 'színész', 'színésznő', 'felnőtt', "mentőautó", "villamos", "fő"]

emb_hu = lang_hu[words_hu]
pca_hu = emb_hu.transform(Pca(2)).plot_interactive(title="Hungarian: PCA")
tsne_hu = emb_hu.transform(Tsne(n_components=2, random_state=0, n_iter=10000, perplexity=2)).plot_interactive(title="Hungarian: t-SNE")
umap_hu = emb_hu.transform(Umap(2)).plot_interactive(title="Hungarian: UMAP")


pca_hu | tsne_hu | umap_hu

#Vektorok

nlp_hu = spacy.load("hu_core_news_lg")
nlp_en = spacy.load("en_core_web_md")
word = nlp_hu("kutya")
print('A kutya szó vektorai: ')
print(word.vector.shape)
word.vector

word = nlp_en('jghfhgfhfhfjhfhf')
print('ajghfhgfhfhfjhfhf szó vektorai: ')
print(word.vector.shape)
print('Szótáron kívüli szó?: ')
print(word[0].is_oov)
word.vector

import numpy as np


def most_similar(word, top_n=5):
    ms = nlp_en.vocab.vectors.most_similar(
        np.asarray([nlp_en.vocab.vectors[nlp_en.vocab.strings[word]]]), n=top_n
    )
    words = [nlp_en.vocab.strings[w] for w in ms[0][0]]
    return words

most_similar("ajax", top_n=10)


# ***Szóvektor modell tanítása***

# *Gensim Word2Vec*

from gensim.models import word2vec
import csv
import random
import pandas as pd
pd.options.display.max_colwidth = 200

import multiprocessing
cores = multiprocessing.cpu_count()
print ('Calculations will be running on {} CPU cores'.format(cores))

# Functions
def recursive_len(item):
    '''Counts the total nr of elements in a list of lists'''
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


# *Korpusz beolvasás*

# * A korpuszt lezárt Stack Overflow kérdésekből állították össze.
# * Csak a kérdések címei szerepelnek
# * A címek a következő előfeldolgozási lépéseken estek át:
#     * tokenizálás
#     * normalizálás: spec karakterek, számok és írásjelek törlése
# * `LineSentence` formátum: 1 mondat (SO kérdés címe) soronként

## Read CSV from url
import csv, urllib.request

url = "https://raw.githubusercontent.com/rfarkas/student_data/main/SO_texts/SO_corpus_lines.csv"
response = urllib.request.urlopen(url)
lines = [l.decode('utf-8') for l in response.readlines()] # response is in bytes that has to be decoded to string encoding
reader = csv.reader(lines)
SO_corpus = list(reader)


# Alapvető információk a korpuszról

print('Lines in corpus:', len(SO_corpus))
print('Full size of corpus, i.e., total number of words:', recursive_len(SO_corpus))
print('Type:', type(SO_corpus))
print('A few random example lines:')
for j in range(5):
    i = random.randint(0, len(SO_corpus))
    print ('Line:', i, ':', SO_corpus[i])


# Gensim - Word2Vec modell építése CBOW módszerrel

# A tanítás egy rejtett réteggel történik, figyelembe veszi a környezetet és a tanult háló rejtett rétegét használja fel szóvektorként. Két fő fajtája:
#
# **CBOW:** A környezet alapján tippeljük meg mi a köztes szó. Bemenet egy adott méretű környezet (pl. 2-2 szó előtte és utána), tanulandó a közöttük levő szó.
#
# **SKIP-GRAM:** Adott szó alapján tippeljük meg a környezet szavait. Bemenet egy szó, tanulandó a környezete pl. 2-2 szó előtte és utána.
#
# Bemenet és kimenet kódolása:
#
# * Bemenet one-hot encodinggal: teljes szótár méretének megfelelő vektor, ahol a bemeneti szónak megfelelő pozícióban 1 szerepel, minden más pozícióban 0.
#
# * Kimenet valószínűségekkel: szótár méretű vektor, a legnagyobb érték mutatja a háló által tippelt szó pozícióját.



# Set values for various training parameters
# reference Google line:
#./word2vec -train corpus_1line.csv -output vectors_cbow.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 12 -binary 1 -iter 15 -alpha 0.025
feature_size = 200    # Word vector dimensionality
window_context = 8    # Context window size
min_word_count = 5    # Minimum word count to filter frequently occured words\n
sample = 1e-4         # Downsample setting for frequent words\n
skipgram = 0          # 0 for CBOW, 1 for skip-gram\n
negative = 25         # value for negative sampling\n
hs = 0                # ({0, 1}, optional) – If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.\n
niter = 15
w2v_model = word2vec.Word2Vec(SO_corpus,
                              size=feature_size,
                              sg=skipgram,
                              negative=negative,
                              hs=hs,
                              window=window_context,
                              min_count=min_word_count,
                              sample=sample,
                              iter=niter,
                              workers=cores)


# A modell tulajdonságai:

print(w2v_model)


# Példák hasonlóságokra: néhány szó és a hozzá 10 leghasonlóbb a korpuszból.


similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=10)]
                  for search_term in ['array', 'loop', 'javascript', 'python', 'digit', 'website', 'recursion','data', 'table', 'ajax']}


res_gensim = pd.DataFrame(similar_words)
res_gensim


w2v_model.wv.get_vector('loop')


# Vizualizáljuk a DataFrameben megjelent szavakat!


from whatlies.language import GensimLanguage


w2v_model.wv.save("word2vec.kv")


lang_w2v = GensimLanguage("word2vec.kv")



words_w2v = set()
for column in res_gensim:
    for word in res_gensim[column]:
        words_w2v.add(word)


emb_w2v = lang_w2v[list(words_w2v)[:40]]
pca_w2v = emb_w2v.transform(Pca(2)).plot_interactive(title="Word2Vec: PCA")
tsne_w2v = emb_w2v.transform(Tsne(n_components=2, random_state=0, n_iter=10000, perplexity=3)).plot_interactive(title="Word2Vec: t-SNE")
umap_w2v = emb_w2v.transform(Umap(2)).plot_interactive(title="Word2Vec: UMAP")



pca_w2v | tsne_w2v | umap_w2v


# ## Floret
#
# https://github.com/explosion/floret/tree/main/python, https://fasttext.cc/docs/en/unsupervised-tutorial.html

# ### Floret modell építése CBOW módszerrel


import floret



#python -m wget('python -m wget https://raw.githubusercontent.com/rfarkas/student_data/main/SO_texts/SO_corpus_lines.csv')


#a csv file megformázása
#sed -i -e 's/,/ /g' SO_corpus_lines.csv")



floret_model = floret.train_unsupervised("SO_corpus_lines_sed.csv",
                                         model="cbow",
                                         mode="floret",
                                         dim=200,
                                         hashCount=2,
                                         bucket=50000,
                                         neg=25,
                                         minn=4,
                                         maxn=5,
                                         thread=cores)


floret_model.save_model("floret.bin")
floret_model.save_vectors("floret.vec")
floret_model.save_floret_vectors("floret.floret")


similar_words = {search_term: [item[1] for item in floret_model.get_nearest_neighbors(search_term, k=10)]
                  for search_term in ['array', 'loop', 'javascript', 'python', 'digit', 'website', 'recursion', 'data', 'table', 'ajax']}

res_floret = pd.DataFrame(similar_words)
res_floret


floret_model.get_word_vector("loop")


from whatlies.language import FloretLanguage


lang_floret = FloretLanguage("floret.bin")


words_floret = set()
for column in res_floret:
    for word in res_floret[column]:
        words_floret.add(word)


emb_floret = lang_floret[list(words_floret)[:40]]
pca_floret = emb_floret.transform(Pca(2)).plot_interactive(title="Floret: PCA")
tsne_floret = emb_floret.transform(Tsne(n_components=2, random_state=0, n_iter=10000, perplexity=3)).plot_interactive(title="Floret: t-SNE")
umap_floret = emb_floret.transform(Umap(2)).plot_interactive(title="Floret: UMAP")


pca_floret | tsne_floret | umap_floret


# # Szöveg-, dokumentumszintű beágyazási módszerek
# A szavak szintje önmagában nagyon kevés információ tartalmaz. Esetenként szükségünk lehet további adatokra, ilyenkor szeretnénk nagyobb egységekre - például mondatokra vagy bekezdésekre - is egy-egy vektort előállítani.
#
# Különböző ötletek jöhetnek ez esetben szóba:
#  * A szövegben található szóvektorok közül válasszunk ki egy reprezentáns elemet
#  * Átlagoljuk ki a szóvektorokat és ez az átlag legyen a szövegszintű vektorunk
#  * Tanítsunk szövegszintű beágyazást

# ### Dokumentumszintű vektorok a spaCy esetén
#
# Egyszerű megoldás: a szóvektorok átlaga.
#
# Érezhető a módszer hátránya: a nyelvi finomságokat könnyen elveszíthetjük, például a szórenddel kifejezett kiemeléseket az átlag nem különbözteti meg mondat szinten.
#
# Ennek ellenére egy gyakorlatban jól alkalmazható megoldás.


# Defináljuk a Dokumentumunkat
doc = nlp_hu("biciklivel megyek boltba")

# Megkeressük a szavak vektorait
wv = []
for tk in doc:
    wv.append(tk.vector)

dv_1 = doc.vector # spaCy dokumentum vektor
dv_2 = np.mean([wv[0], wv[1], wv[2]], axis=0) # numpy-val kiszámítjuk a szóvektorok átlagát

# Megnézzük, hogy a különbség 0-e, ha igen, akkor bizonyítottuk, hogy a spaCy is így számol dokumentum vektort
all(v == 0 for v in dv_1 - dv_2)


doc1 = nlp_hu("villamossal megyek a térre")
doc2 = nlp_hu("a térre villamsosal megyek")
print('Dokumentumok hasonlósága:', doc1.similarity(doc2))


# ## Tanulás alapú megoldások
#
# ### Bert és egyéb Transformer-hálók
# - Léteznek már magyar nyelvre is dokumentszintű reprezentáló előtanított modellek, ilyen a [huBERT](https://huggingface.co/SZTAKI-HLT/hubert-base-cc) is, amit például a HuSpaCy [`hu_core_news_trf`](https://huggingface.co/huspacy/hu_core_news_trf) modellje használ is. Ezzel rendkívűl pontos előfeldolgozó rendszereket lehet finomhangolni, hátránya persze a nagyon sok millió paraméternek köszönhetően, hogy lassú, ami sok esetben hátrány, de alkalmazás válogatja. Aki szeretne mélyebben megismerkedni ezen neuronhálókkal és még sok egyébbel, azoknak ajánljuk Berend Gábor óráját a [Számítógépes szemantikát](https://github.com/begab/compsem/).
#
# ### Doc2Vec
# - És ahogyan szó volt, léteznek jóval egyszerűbb megoldások is, mint a Doc2Vec.
#

# # Dokumentumosztályozás szóbeágyazásra építve


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
import tensorflow as tf
import multiprocessing
import numpy as np
import os


# ## Töltsük le az IMDB reviews adathalmazt

# A train_reviews és test_reviews ndarray objektumok, melyek integer-eket tartalmazó listákból állnak. 25000 ilyen listát találhatunk mindkét objektumban. Egy-egy integer-eket tartalmazó lista egy-egy review. Az integer-ek egyedi azonosítók, melyek a szavakat kódolják. A train_labels és test_labels szintén ndarray objektumok, melyek a review-k címkéit tartalmazzák. Minden egyes review-t pozitív (1) vagy negatív (0) kategóriába sorolunk.

imdb = tf.keras.datasets.imdb
(train_reviews, train_labels), (test_reviews, test_labels) = imdb.load_data()


train_reviews[0]



train_labels[0]


# Készítsük el az szavakat reprezentáló integer-ek dekódolásához a szótárunkat. Ehhez használni tudjuk az imdb objektum get_word_index() metódusát, ami egy dictionary-t ad vissza, melyben a szavak string alakja a kulcs, az integer alakja pedig az érték. A dictionary elejéhez hozzáadunk néhány tag-et.

vocab = imdb.get_word_index()
vocab = {k:(v + 3) for k, v in vocab.items()}
vocab["<PAD>"] = 0
vocab["<START>"] = 1
vocab["<UNK>"] = 2
vocab["<UNUSED>"] = 3


vocab["brilliant"]


# Készítsünk inverz szótárat, amelyben a szavak integer alakja a kulcs, a string alakja pedig az érték.

vocab_inv = dict([(value, key) for (key, value) in vocab.items()])


vocab_inv[530]


# Definiáljunk egy decode_review() metódust, mely a review-k integer reprezentációját string reprezentációvá alakítja.

def decode_review(review):
    return [vocab_inv.get(i, "?") for i in review]

decode_review(train_reviews[0])


# ## Doc2Vec dokumentumbeágyazások létrehozása

# A dokumentumvektorok elkészítéséhez a gensim csomagot fogjuk használni. Alakítsuk a review-kat TaggedDocument objektumokká.

reviews = np.concatenate((train_reviews, test_reviews))
docs = [TaggedDocument(decode_review(review), [i]) for i, review in enumerate(reviews)]

class Doc2VecCallback(CallbackAny2Vec):
    def __init__(self, epochs):
        self.prog_bar = tf.keras.utils.Progbar(epochs)
        self.epoch = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        self.prog_bar.update(self.epoch)


# A Doc2Vec osztály elkészíti a review-k dokumentum vektorait.

d2v_model = Doc2Vec(docs, dm=0, min_count=2, vector_size=100, hs=0, negative=5, epochs=5,
                    callbacks=[Doc2VecCallback(5)], sample=0, workers=multiprocessing.cpu_count())

# fname = '/content/d2v_model'
# d2v_model = Doc2Vec.load(fname)  # you can continue training with the loaded model!


# Nyerjük ki a modellből a train_reviews és test_reviews adataink vektoros reprezentációit.

embdgs = d2v_model.docvecs.vectors_docs
train_embdgs, test_embdgs = np.split(embdgs, [25000])


train_embdgs[0]


# ## Osztályozás Doc2Vec beágyazások alapján

model_1 = tf.keras.Sequential()
model_1.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model_1.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.01), loss='binary_crossentropy',
              metrics=['accuracy'])

model_1.fit(train_embdgs, train_labels, batch_size=64, epochs=50, shuffle=True)


model_1.evaluate(test_embdgs, test_labels)


# Hozzunk létre egy fokkal összetettebb neuronhálót.

model_2 = tf.keras.Sequential()
model_2.add(tf.keras.layers.Dense(50, activation="relu", input_shape=(100, )))
model_2.add(tf.keras.layers.Dropout(0.3))
model_2.add(tf.keras.layers.Dense(50, activation="relu"))
model_2.add(tf.keras.layers.Dropout(0.2))
model_2.add(tf.keras.layers.Dense(50, activation="relu"))
model_2.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model_2.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.01), loss='binary_crossentropy',
              metrics=['accuracy'])

model_2.fit(train_embdgs, train_labels, batch_size=64, epochs=50, shuffle=True)


model_2.evaluate(test_embdgs, test_labels)


# ## Példa Keras kód CBOW beágyazás tanításához

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Lambda, dot, Input, Reshape
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# example CBOW architecture in Keras

vocab=np.zeros(10000)

vocab_size=len(vocab)
embed_size=200
window_size=8

cbow = Sequential()

cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))

cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())

# visualize model structure

SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False,
                 rankdir='TB').create(prog='dot', format='svg'))



