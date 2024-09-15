from lib import BeepClassifier

DATA_DIRECTORY = "dataset"


def main():
    classifier = BeepClassifier()
    classifier.train(DATA_DIRECTORY)


if __name__ == "__main__":
    main()
