import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class EventClusterizer:
    def __init__(self, cluster_num=20):
        self.cluster_num = cluster_num

    def process(self):
        specs_df = pd.read_csv('/code/data/raw/specs.csv')
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(specs_df['info'].values)
        clusters = KMeans(
            n_clusters=self.cluster_num,
            random_state=77
            ).fit_predict(X.toarray())
        clusters = ['event_' + str(i) for i in clusters]
        specs_df['clusters'] = clusters

        return specs_df.set_index('event_id')['clusters']
