import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import math
from sklearn.linear_model import SGDClassifier
import numpy as np
import csv
from preprocess_functions import tfidf

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_n = list(train["name"])
train_d = list(train["description"])
x_train = [train_n[i] + " " + train_d[i] for i in range(len(train_n))]

test_n = list(test["name"])
test_d = list(test["description"])
x_test = [test_n[i] + " " + test_d[i] for i in range(len(test_n))]

vect_traintest = x_train + x_test

vectorized_corpus = tfidf(vect_traintest)

x_train_vec = vectorized_corpus[:408]
x_test_vec = vectorized_corpus[408:]

categories = {'Агент': 0,
 'Учитель': 1,
 'Дизайнер': 2,
 'IT': 3,
 'Инженер': 4,
 'Специалист': 5,
 'other': 6,
 'Рабочий': 7,
 'СМИ': 8,
 'Менеджер': 9,
 'Право': 10,
 'Искусство': 11,
 'Врач': 12}

y_train = list(train["category"])
for i in range(len(y_train)):
    y_train[i] = categories[y_train[i]]

x_train_vec = np.array(x_train_vec)
y_train = np.array(y_train)
x_test_vec = np.array(x_test_vec)

model = SGDClassifier(random_state=42)

model.fit(x_train_vec, y_train)

answer = model.predict(x_test_vec)

categories_list = list(categories.items())
answer = list(answer)
for i in range(len(answer)):
    answer[i] = categories_list[answer[i]][0]

with open("answer.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter = ",", lineterminator="\r")
    file_writer.writerow(["id", "category"])
    for i in range(106):
        file_writer.writerow([str(i), answer[i]])
