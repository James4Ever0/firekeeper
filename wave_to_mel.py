from lib import MelBeepClassifier


def main():
    # source = 'output.wav'
    source = 'dataset/office_beep_1.wav'
    #source = 'dataset/beep_1.wav'
    classifier = MelBeepClassifier()
    label = classifier.predict(source)
    print('Label:', label)

if __name__ == "__main__":
    main()