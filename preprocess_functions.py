from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import math

stemmer = SnowballStemmer(language = "russian")
stop_words = stopwords.words("russian")

def get_punct(corpus: list) -> list:
    corpus_symbols = set(",".join(corpus))
    symbols = [i for i in corpus_symbols if i.isalnum() != 1 and i != " "]
    return symbols

def del_punct(corpus: list) -> list:
    symbols = get_punct(corpus)
    for i in range(len(corpus)):
        doc = [s.lower() if s not in symbols else " " for s in corpus[i]]
        corpus[i] = "".join(doc)
    return corpus

def stem_corpus(corpus: list) -> list:
    for i in range(len(corpus)):
        doc = [stemmer.stem(w) for w in corpus[i].split()]
        corpus[i] = " ".join(doc)
    return corpus

def del_stop(corpus: list) -> list:
    for i in range(len(corpus)):
        doc = [w for w in corpus[i].split() if w not in stop_words]
        corpus[i] = " ".join(doc)
    return corpus

def tfidf(corpus: list) -> list:
    vect_corpus = corpus.copy()
    vect_corpus = del_punct(vect_corpus)
    vect_corpus = del_stop(vect_corpus)
    vect_corpus = stem_corpus(vect_corpus)
    
    num_words = {}
    cnt = 0
    
    for doc in vect_corpus:
        for word in doc.split():
            if word in num_words:
                num_words[word][cnt] += 1
            else:
                num_words.update({word: [0]*len(vect_corpus)})
                num_words[word][cnt] += 1
        cnt += 1
        
    tf = {}
    for word in num_words.keys():
        tf.update({word: [0]*len(vect_corpus)})
        for i in range(len(vect_corpus)):
            tf[word][i] = num_words[word][i]/len(vect_corpus[i])
            
    idf = {}
    for word in num_words.keys():
        idf_w = math.log(len(vect_corpus) 
                / (len(num_words[word]) - num_words[word].count(0)))
        idf.update({word: idf_w})
        
    for i in range(len(vect_corpus)):
        new_doc = []
        for word in num_words.keys():
            new_doc.append(tf[word][i]*idf[word])
        vect_corpus[i] = new_doc
        
    return vect_corpus