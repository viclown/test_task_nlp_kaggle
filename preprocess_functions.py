from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import math

def del_punct(corpus):
    symbols = [".", ",", "/", "?", ">", "<", "!", "", ":", "-", "—", "«", "»"]
    for i in range(len(corpus)):
        doc = [s.lower() if s not in symbols else " " for s in corpus[i]]
        corpus[i] = "".join(doc)
    return corpus

def stem_corpus(corpus):
    stemmer = SnowballStemmer(language = "russian")
    for i in range(len(corpus)):
        doc = [stemmer.stem(w) for w in corpus[i].split()]
        corpus[i] = " ".join(doc)
    return corpus

def del_stop(corpus):
    stop_words = stopwords.words("russian")
    for i in range(len(corpus)):
        doc = [w for w in corpus[i].split() if w not in stop_words]
        corpus[i] = " ".join(doc)
    return corpus

def tfidf(corpus):
    corpus = del_punct(corpus)
    corpus = del_stop(corpus)
    corpus = stem_corpus(corpus)
    
    num_words = {}
    cnt = 0
    
    for doc in corpus:
        for word in doc.split():
            if word in num_words:
                num_words[word][cnt] += 1
            else:
                num_words.update({word: [0]*len(corpus)})
                num_words[word][cnt] += 1
        cnt += 1
        
    tf = {}
    for word in num_words.keys():
        tf.update({word: [0]*len(corpus)})
        for i in range(len(corpus)):
            tf[word][i] = num_words[word][i]/len(corpus[i])
            
    idf = {}
    for word in num_words.keys():
        idf_w = math.log(len(corpus) 
                / (len(num_words[word]) - num_words[word].count(0)))
        idf.update({word: idf_w})
        
    for i in range(len(corpus)):
        new_doc = []
        for word in num_words.keys():
            new_doc.append(tf[word][i]*idf[word])
        corpus[i] = new_doc
