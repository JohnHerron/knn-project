from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.utils import Bunch
import numpy as np
import csv
from plot_cv_indices import plot_cv_indices, cmap_cv

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)


def main():
    seed_data = load_seeds()

    # Apply PCA
    pipeline = make_pipeline(StandardScaler(), PCA(n_components='mle'))
    x_pca = pipeline.fit_transform(seed_data.data)

    print(x_pca)

    # knn classification
    y = seed_data.target

    # Cross-validation strategies
    cvs = [StratifiedKFold, StratifiedShuffleSplit, KFold, TimeSeriesSplit]

    for cv in cvs:
        fig, ax = plt.subplots(figsize=(5, 3))
        plot_cv_indices(cv(n_splits=5), x_pca, y, ax, n_splits=5)
        ax.legend(
            [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
            ["Testing set", "Training set"],
            loc=(1.02, 0.8),
        )
        ax.set_title(cv.__name__)
        # Make the legend fit
        plt.tight_layout()
        fig.subplots_adjust(right=0.7)
        plt.show()

    # manual k-15 test
    classifier_pipeline = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    y_pred = cross_val_predict(classifier_pipeline, x_pca, y, cv=cv)
    print(mean_squared_error(y, y_pred))
    print(r2_score(y, y_pred))

    # test for k, which produces lowest error?
    # mean squared error
    error = []
    for k in range(1, 51):
        cv = KFold(n_splits=5, random_state=0, shuffle=True)
        classifier_pipeline = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        y_pred = cross_val_predict(classifier_pipeline, x_pca, y, cv=cv)
        error.append(mean_squared_error(y, y_pred))
    plt.plot(range(1, 51), error)
    plt.show()

    # r squared error
    error = []
    for k in range(1, 51):
        classifier_pipeline = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        y_pred = cross_val_predict(classifier_pipeline, x_pca, y, cv=5)
        error.append(r2_score(y, y_pred))
    plt.plot(range(1, 51), error)
    plt.show()


def load_seeds():
    data_file_name = "seeds.csv"
    data, target, target_names = load_csv_data(data_file_name=data_file_name)
    feature_names = [
        "area A",
        "perimeter P",
        "compactness C = 4*pi*A/P^2",
        "length of kernel",
        "width of kernel",
        "asymmetry coefficient",
        "length of kernel groove"
    ]

    return Bunch(
        data=data,
        target=target,
        target_names=target_names,
        feature_names=feature_names,
        filename=data_file_name
    )


def load_csv_data(data_file_name):
    with open(data_file_name, "r", encoding="utf-8") as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)

        return data, target, target_names


if __name__ == "__main__":
    main()
