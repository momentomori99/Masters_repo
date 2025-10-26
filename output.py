import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load output and coerce into a (n_samples, 4000) array
output = np.load('data/X_output2.npy', allow_pickle=True)
y = np.load('data/iris_y.npy', allow_pickle=True)
X = np.array(output)

original_X = np.load('data/iris_X_rates.npy', allow_pickle=True)
original_y = np.load('data/iris_y.npy', allow_pickle=True)

original_X = np.array(original_X)
original_y = np.array(original_y)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings


def safe_pca_components(X: np.ndarray, max_components: int = 50) -> int:
    n_samples, n_features = X.shape
    # PCA components cannot exceed min(n_samples, n_features)
    return max(1, min(max_components, min(n_samples, n_features) - 1))


def compute_explained_variance_ratio(X: np.ndarray, n_components: int) -> float:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    return float(np.sum(pca.explained_variance_ratio_))


def evaluate_sklearn_models(X: np.ndarray, y: np.ndarray, dataset_name: str, n_components: int) -> None:
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RidgeClassifier': RidgeClassifier(),
        'LinearSVM': LinearSVC(max_iter=5000),
        'RBF-SVM': SVC(kernel='rbf', gamma='scale', C=1.0),
        'KNN-5': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Sklearn-MLP': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                                     max_iter=500, random_state=42)
    }

    print(f"\n=== {dataset_name}: PCA(n_components={n_components}) + Models (5-fold CV) ===")
    for name, estimator in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components),
                estimator
            )
            cvres = cross_validate(pipeline, X, y, cv=5, scoring='accuracy', return_train_score=True)
            test_scores = cvres['test_score']
            train_scores = cvres['train_score']
        print(f"{name:16s} | cv acc: {test_scores.mean():.4f} ± {test_scores.std():.4f} | train acc: {train_scores.mean():.4f} ± {train_scores.std():.4f}")


def evaluate_tf_mlp_cv(X: np.ndarray, y: np.ndarray, dataset_name: str, n_components: int) -> None:
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except Exception:
        print(f"TensorFlow not available; skipping TF MLP for {dataset_name}.")
        return

    # Prepare CV with in-fold scaling + PCA to avoid leakage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    fold_train_scores = []
    num_classes = int(len(np.unique(y)))

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        X_train = scaler.fit_transform(X_train)
        X_train = pca.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = pca.transform(X_test)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_components,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[es], verbose=0)
        # Train accuracy (on the training fold)
        _, acc_train = model.evaluate(X_train, y_train, verbose=0)
        # Test accuracy (on the fold's holdout)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        fold_train_scores.append(acc_train)
        fold_scores.append(acc)

    fold_scores = np.array(fold_scores)
    print(f"TF-MLP         | cv acc: {fold_scores.mean():.4f} ± {fold_scores.std():.4f} | train acc: {np.mean(fold_train_scores):.4f} ± {np.std(fold_train_scores):.4f}")


def evaluate_sklearn_models_no_pca(X: np.ndarray, y: np.ndarray, dataset_name: str) -> None:
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RidgeClassifier': RidgeClassifier(),
        'LinearSVM': LinearSVC(max_iter=5000),
        'RBF-SVM': SVC(kernel='rbf', gamma='scale', C=1.0),
        'KNN-5': KNeighborsClassifier(n_neighbors=5),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Sklearn-MLP': MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                                     max_iter=500, random_state=42)
    }

    print(f"\n=== {dataset_name}: No PCA + Models (5-fold CV) ===")
    for name, estimator in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline = make_pipeline(
                StandardScaler(),
                estimator
            )
            cvres = cross_validate(pipeline, X, y, cv=5, scoring='accuracy', return_train_score=True)
            test_scores = cvres['test_score']
            train_scores = cvres['train_score']
        print(f"{name:16s} | cv acc: {test_scores.mean():.4f} ± {test_scores.std():.4f} | train acc: {train_scores.mean():.4f} ± {train_scores.std():.4f}")


def evaluate_tf_mlp_cv_no_pca(X: np.ndarray, y: np.ndarray, dataset_name: str) -> None:
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except Exception:
        print(f"TensorFlow not available; skipping TF MLP (no PCA) for {dataset_name}.")
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    fold_train_scores = []
    num_classes = int(len(np.unique(y)))

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[es], verbose=0)
        _, acc_train = model.evaluate(X_train, y_train, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        fold_train_scores.append(acc_train)
        fold_scores.append(acc)

    fold_scores = np.array(fold_scores)
    print(f"TF-MLP (no PCA) | cv acc: {fold_scores.mean():.4f} ± {fold_scores.std():.4f} | train acc: {np.mean(fold_train_scores):.4f} ± {np.std(fold_train_scores):.4f}")


def main() -> None:
    np.random.seed(42)

    # Basic checks
    if not np.array_equal(y, original_y):
        print("Warning: y and original_y differ; proceeding with 'y' for both evaluations.")

    # Decide PCA components per dataset
    n_components_X = safe_pca_components(X, max_components=50)
    n_components_orig = safe_pca_components(original_X, max_components=50)

    # Report explained variance (fit on full data, for info only)
    evr_X = compute_explained_variance_ratio(X, n_components_X)
    evr_orig = compute_explained_variance_ratio(original_X, n_components_orig)
    print(f"Output X: PCA n_components={n_components_X}, variance retained={evr_X:.4f}")
    print(f"Original X: PCA n_components={n_components_orig}, variance retained={evr_orig:.4f}")

    # Evaluate classic sklearn models
    evaluate_sklearn_models(X, y, dataset_name='Output X', n_components=n_components_X)
    evaluate_sklearn_models(original_X, y, dataset_name='Original X', n_components=n_components_orig)

    # Evaluate TensorFlow MLP (optional)
    print("\n=== Deep Learning (TensorFlow MLP) ===")
    print("Output X:")
    evaluate_tf_mlp_cv(X, y, dataset_name='Output X', n_components=n_components_X)
    print("Original X:")
    evaluate_tf_mlp_cv(original_X, y, dataset_name='Original X', n_components=n_components_orig)

    # Evaluate Original X without PCA
    evaluate_sklearn_models_no_pca(original_X, y, dataset_name='Original X')
    print("Deep Learning (TensorFlow MLP) — No PCA:")
    evaluate_tf_mlp_cv_no_pca(original_X, y, dataset_name='Original X')


if __name__ == '__main__':
    main()
