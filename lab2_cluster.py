import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from lab2_read_txt import *
from nltk.stem.snowball import SnowballStemmer

print(str(len(all_wall)) + ' запросов считано')
stemmer = SnowballStemmer("russian")
# nltk.download()

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# Создаем словари (массивы) из полученных основ
totalvocab_stem = []
totalvocab_token = []
for i in all_wall:
    allwords_stemmed = token_and_stem(i)
    totalvocab_stem.extend(allwords_stemmed)

    allwords_tokenized = token_only(i)
    totalvocab_token.extend(allwords_tokenized)

print("Слова после нормализации:")
print(totalvocab_stem[100:150])


stopwords = nltk.corpus.stopwords.words('russian')
#можно расширить список стоп-слов
stopwords.extend(['что', 'это', 'так', 'вот',
'быть', 'как', 'в', 'к', 'на', 'бол', 'оп', 'больш', 'будт', 'быт', 'вед', 'впроч', 'всег', 'всегд', 'даж', 'друг', 'е',
'ег', 'ем', 'есл', 'ест', 'ещ', 'зач', 'зде', 'ил', 'иногд', 'когд', 'конечн', 'куд', 'лучш', 'межд', 'мен', 'мног',
'мо', 'можн', 'нег', 'нельз', 'нибуд', 'никогд', 'нич', 'опя', 'посл', 'пот', 'почт', 'разв', 'сво', 'себ',
'совс', 'теб', 'тепер', 'тог', 'тогд', 'тож', 'тольк', 'хорош', 'хот', 'чег', 'чут', 'эт'])
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000, min_df=0.01, stop_words=stopwords, use_idf=True,
                                   tokenizer=token_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(all_wall)
print("Размерность матрицы TF-IDF: ", tfidf_matrix.shape)

num_clusters = 5
# Метод к-средних - KMeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print("Метод к-средних - KMeans:\n", clusters)

# MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(init='random', n_clusters=num_clusters) #(init='k-means++', ‘random’ or an ndarray)
mbk.fit_transform(tfidf_matrix)
mbk.fit(tfidf_matrix)
miniclusters = mbk.labels_.tolist()
print("MiniBatchKMeans:")
print (mbk.labels_.tolist())
# DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(tfidf_matrix)
labels = db.labels_
print("DBSCAN:")
print(labels.shape)
print(labels.tolist())
# Аггломеративная класстеризация
from sklearn.cluster import AgglomerativeClustering
agglo1 = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean')

answer = agglo1.fit_predict(tfidf_matrix.toarray())
print("Аггломеративная класстеризация:")
print(answer.shape)
print(answer.tolist())

#k-means
clusterkm = km.labels_.tolist()
#minikmeans
clustermbk = mbk.labels_.tolist()
#dbscan
clusters3 = labels
 #agglo
#clusters4 = answer.tolist()
frame = pd.DataFrame(all_wall, index = [clusterkm])
#k-means
out = { 'title': all_wall, 'cluster': clusterkm }
frame1 = pd.DataFrame(out, index = [clusterkm], columns = ['title', 'cluster'])
#mini
out = { 'title': all_wall, 'cluster': clustermbk }
frame_minik = pd.DataFrame(out, index = [clustermbk], columns = ['title', 'cluster'])
print("k-means\n", frame1['cluster'].value_counts())
print("minikmeans\n", frame_minik['cluster'].value_counts())

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print("dist.shape: ", dist.shape)

# Метод главных компонент - PCA
from sklearn.decomposition import IncrementalPCA
icpa = IncrementalPCA(n_components=2, batch_size=16)
icpa.fit(dist)
demo2 = icpa.transform(dist)
xs, ys = demo2[:, 0], demo2[:, 1]
# PCA 3D
from sklearn.decomposition import IncrementalPCA
icpa = IncrementalPCA(n_components=3, batch_size=16)
icpa.fit(dist)
ddd = icpa.transform(dist)
xs, ys, zs = ddd[:, 0], ddd[:, 1], ddd[:, 2]
#Можно сразу примерно посмотреть, что получится в итоге
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()