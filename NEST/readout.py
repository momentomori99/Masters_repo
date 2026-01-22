from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


class Readout:
    def __init__(self, data, target):
        self.X = data
        self.y = target
        self.model = LogisticRegression(max_iter=1000)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self):
        self.split_data()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        print(accuracy_score(self.y_test, y_pred))

    def cross_validation(self, cv_fold = 5):
        return self.cross_validation_pca(n_components=None, cv_fold=cv_fold)

    def cross_validation_pca(self, n_components=50, cv_fold=5):
        """
        Cross-validated accuracy with optional PCA.

        Parameters
        ----------
        n_components : int | None
            Number of principal components to keep. If None, PCA is skipped.
        cv_fold : int
            Number of CV folds.
        """
        steps = [('scaler', StandardScaler())]
        if n_components is not None:
            steps.append(('pca', PCA(n_components=n_components)))
        steps.append(("clf", LogisticRegression(max_iter=1000)))

        model = Pipeline(steps)
        cv = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=42)
        scores = cross_validate(model, self.X, self.y, cv=cv, scoring='accuracy', return_train_score=True)
        summary = ""
        summary += f"Fold accuracies: {scores['test_score']}\n"
        summary += f"Mean accuracy: {scores['test_score'].mean():.4f} ± {scores['test_score'].std():.4f}\n"
        summary += f"Train accuracies: {scores['train_score']}\n"
        summary += f"Mean train accuracy: {scores['train_score'].mean():.4f} ± {scores['train_score'].std():.4f}\n"
        print(summary)
        return summary