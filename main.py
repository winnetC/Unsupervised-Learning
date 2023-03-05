import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

NewsArticles = pd.read_csv('NewsArticles.csv')

summary_list=[]

for x in NewsArticles['summary']:
  summary_list.append(x)

# print(len(summary_list))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(summary_list)
true_k = 3

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=30, n_init=1)
model.fit(X)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     print("\n")

labels = model.fit(X)
# labels.labels_

NewsArticles['cluster_label'] = labels.labels_
# print(NewsArticles.head(5))

import streamlit as st

header = st.container()
# info = st.container()
dataset = st.container()
links = st.container()

with header:
  st.header('Q2')
  # st.text('Question...')

# with info:
#   st.header('the question')

with dataset:
  # st.header('dataframe of info extracted from scrawling and scraping the web')
  st.text('Crawled and scraped news articles dataset')

  # news_article = pd.read_csv('folder/news_article.csv')
  st.write(NewsArticles)

with links:
  st.header('Clustered links')
  st.text('')
  st.text('Cluster Group 0')

  for index, p in enumerate(NewsArticles.cluster_label):
      if p == 0:
          y = index
          z = NewsArticles['link'].values[y]
          st.write("[link]z")

  st.text('')

  st.text('Cluster Group 1')

  for index, p in enumerate(NewsArticles.cluster_label):
      if p == 1:
          y = index
          z = NewsArticles['link'].values[y]
          st.write("[link]z")

  st.text('')

  st.text('Cluster Group 2')

  for index, p in enumerate(NewsArticles.cluster_label):
      if p == 2:
          y = index
          z = NewsArticles['link'].values[y]
          st.write("[link]z")
