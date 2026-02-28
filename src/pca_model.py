from sklearn.decomposition import PCA

class PCAModel:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components = self.n_components)

    def fit(self, X):
        self.pca.fit(X)

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

    def explained_variance(self):
        return self.pca.explained_variance_ratio_

