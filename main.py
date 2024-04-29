from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import numpy as np
import csv


def main():
    seed_data = load_seeds()

    # knn classification
    x = seed_data.data
    y = seed_data.target
    classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    #cv = KFold(n_splits=5, random_state=0, shuffle=True)
    y_pred = cross_val_predict(classifier_pipeline, x, y, cv=5)
    print(mean_squared_error(y, y_pred))
    print(r2_score(y, y_pred))

    # test for k, which produces lowest error?
    error = []
    for k in range(1,51):
        classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        y_pred = cross_val_predict(classifier_pipeline, x, y, cv=5)
        error.append(mean_squared_error(y,y_pred))
    plt.plot(range(1,51),error)
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
        target = target,
        target_names = target_names,
        feature_names= feature_names,
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