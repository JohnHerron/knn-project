from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass

def main():
    seed_data = load_seeds()
    # TODO: knn classification

def load_seeds():
    data = []
    target = []

    # TODO: parse text file

    return seed_data(data, target)

@dataclass
class seed_data:
    data: list
    target = list

if __name__ == "__main__":
    main()