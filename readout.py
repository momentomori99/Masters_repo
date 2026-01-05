from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


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
        model = Pipeline([
            ('scaler', StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))])

        cv = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=42)
        scores = cross_validate(model, self.X, self.y, cv=cv, scoring='accuracy', return_train_score=True)
        print(f"Fold accuracies: {scores['test_score']}")
        print(f"Mean accuracy: {scores['test_score'].mean():.4f} ± {scores['test_score'].std():.4f}")
        print(f"Train accuracies: {scores['train_score']}")
        print(f"Mean train accuracy: {scores['train_score'].mean():.4f} ± {scores['train_score'].std():.4f}")