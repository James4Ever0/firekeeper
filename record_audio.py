from lib import AudioRecorder

WAVE_OUTPUT_FILENAME = "output.wav"


def main():
    recorder = AudioRecorder()
    recorder.record(WAVE_OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
