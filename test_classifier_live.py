from lib import BeepClassifier


def main():
    classifier = BeepClassifier()
    counter = 0
    beep_counter = 0
    while True:
        beep = classifier.predict_live()
        if int(beep) == 1:
            beep_counter += 1
        print(f"Round #{counter}: {beep_counter} beeps")
        counter += 1


if __name__ == "__main__":
    main()
