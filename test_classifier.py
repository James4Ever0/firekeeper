from lib import BeepClassifier
from record_audio import WAVE_OUTPUT_FILENAME


def main():
    classifier = BeepClassifier()
    classifier.predict(WAVE_OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
