from lib import AudioRecorderClient
import os
import json
WAVE_OUTPUT_FILENAME = "output.wav"
recorder_server_host = json.load(open(".secrets.json"))['recorder_server_host']

HOST = os.environ.get("AUDIO_RECORDER_SERVER_HOST", recorder_server_host)


def main():
    recorder = AudioRecorderClient(HOST)
    recorder.record(WAVE_OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
