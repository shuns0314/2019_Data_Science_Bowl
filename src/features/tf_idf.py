import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class EventClusterizer:
    def __init__(self, info_cluster_num=10, args_cluster_num=8):
        self.info_cluster_num = info_cluster_num
        self.args_cluster_num = args_cluster_num
        self.specs_df = pd.read_csv('/code/data/raw/specs.csv')

    def process(self):
        self.specs_df['info_clusters'] = self.vectorize(
            columns='info', cluster_num=self.info_cluster_num)
        self.specs_df['args_clusters'] = self.vectorize(
            columns='args',  cluster_num=self.args_cluster_num)
        self.specs_df = self.specs_df.set_index('event_id')
        return self.specs_df[['info_clusters', 'args_clusters']]

    def vectorize(self, columns, cluster_num):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.specs_df[columns].values)
        clusters = KMeans(
            n_clusters=cluster_num,
            random_state=77,
            ).fit_predict(X.toarray())
        clusters = [f'{columns}_' + str(i) for i in clusters]
        return clusters
