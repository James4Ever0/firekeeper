import abc
import http
import json
import os
import tempfile
import traceback
import wave
from enum import Enum

import fastapi
import joblib
import librosa
import numpy as np
import pyaudio
import requests
import uvicorn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ENCODING = "utf-8"
SIGNAL_SEND_URL = json.load(open(".secrets.json"))["signal_send_url"]

AUDIO_RECORDER_SERVER_HOST = "0.0.0.0"
AUDIO_RECORDER_SERVER_PORT = 8225
AUDIO_RECORDER_SERVER_ENDPOINT = "/record"
TIMEOUT_SECONDS = 10


class Severity(Enum):
    MODERATE = "3"
    WARNING = "4"
    ALERT = "5"


def process_audio_file(file_path):
    y, sr = librosa.load(file_path)
    # Apply Fourier Transform
    D = np.abs(librosa.stft(y))
    return D


class AbstractAudioRecorder(abc.ABC):
    @abc.abstractmethod
    def record(self, output_path: str): ...
class AudioRecorder(AbstractAudioRecorder):
    def __init__(
        self,
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        chunk=1024,
        record_seconds=5,
    ) -> None:
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds

    def record(self, output_path: str):

        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("Recording...")

        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Saving audio to:", output_path)
        wf = wave.open(output_path, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(frames))
        wf.close()


class AudioRecorderServer(AudioRecorder):
    def __init__(
        self, host=AUDIO_RECORDER_SERVER_HOST, port=AUDIO_RECORDER_SERVER_PORT
    ):
        self.host = host
        self.port = port
        self.app = fastapi.FastAPI()

    def record_and_reply_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "audio.wav")
            self.record(filepath)
            with open(filepath, "rb") as f:
                content = f.read()
                return fastapi.Response(
                    content=content, media_type="application/octet-stream"
                )

    def main(self):
        self.app.get(AUDIO_RECORDER_SERVER_ENDPOINT)(self.record_and_reply_audio)
        uvicorn.run(self.app, host=self.host, port=self.port)


class AudioRecorderClient(AbstractAudioRecorder):
    def __init__(
        self, host: str, port=AUDIO_RECORDER_SERVER_PORT, timeout=TIMEOUT_SECONDS
    ):
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}{AUDIO_RECORDER_SERVER_ENDPOINT}"
        self.timeout = timeout

    def record(self, output_path: str):
        r = requests.get(self.endpoint, timeout=self.timeout)
        assert (
            r.status_code == http.HTTPStatus.OK
        ), f"Invalid status code: {r.status_code}"
        content = r.content
        print("Writing recorded audio to:", output_path)
        with open(output_path, "wb") as f:
            f.write(content)


class BeepClassifier:
    def __init__(self, label: str = "beep", model_path: str = "beep_classifier.pkl"):
        self.recorder = AudioRecorder()
        self.label = label
        self.model_path = model_path
        if os.path.exists(model_path):
            self.load_model()
        else:
            print("Saved model not found at:", self.model_path)
            self.classifier = None

    def train(self, dataset_directory: str, accuracy_threshold=0.9):
        # Load and process audio data
        audio_files = [
            f"{dataset_directory}/{it}" for it in os.listdir(dataset_directory)
        ]
        labels = [int(self.label in it) for it in audio_files]
        print("Labels:", labels)

        # Process audio files and extract features
        X = np.array([process_audio_file(file) for file in audio_files])
        y = labels

        # Reshape the features for training
        X = X.reshape(X.shape[0], -1)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,  # random_state=42
        )

        # Initialize neural network classifier
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

        # Train the classifier
        clf.fit(X_train, y_train)
        # clf.fit(X, y)

        # Evaluate the classifier
        accuracy = clf.score(X_test, y_test)
        # accuracy = clf.score(X, y)
        print(f"Accuracy: {accuracy}")

        if accuracy >= accuracy_threshold:
            self.classifier = clf
            self.save_model()
        else:
            print("Model accuracy smaller than threshold:", accuracy_threshold)
            print("Not saving model")

    def save_model(self):
        print("Saving model to:", self.model_path)
        joblib.dump(self.classifier, self.model_path)

    def load_model(self):
        print("Loading model from:", self.model_path)
        self.classifier = joblib.load(self.model_path)

    def predict(self, filepath: str):
        print("Predicting from file:", filepath)
        X = np.array([process_audio_file(filepath)])
        X = X.reshape(X.shape[0], -1)
        y = int(self.classifier.predict(X)[0])
        print("Predicted label:", y)
        return y

    def predict_live(self):
        print("Performing live prediction")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_filepath = os.path.join(tmpdir, "record.wav")
            self.recorder.record(audio_filepath)
            ret = self.predict(audio_filepath)
            return ret


class RemoteBeepClassifier(BeepClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.recorder = AudioRecorderClient(*args, **kwargs)


class MessageSender:
    def __init__(self, url=SIGNAL_SEND_URL, encoding=ENCODING, timeout=TIMEOUT_SECONDS):
        self.url = url
        self.encoding = encoding
        self.timeout = timeout

    def send(self, message: str, severity: Severity):
        try:
            requests.post(
                SIGNAL_SEND_URL,
                data=message.encode(encoding=self.encoding),
                headers={"Priority": severity},
                timeout=self.timeout,
            )
        except:
            traceback.print_exc()
            print("Failed to post the message")


class FireKeeper:
    def __init__(self, beep_threshold=3) -> None:
        self.beep_classifier = BeepClassifier()
        self.beep_threshold = beep_threshold
        self.beep_counter = 0
        self.message_sender = MessageSender()

    def handle_fire(self):
        self.send_fire_notification()
        self.turn_off_devices()

    def send_fire_notification(self):
        message = (
            f"Smoke detected for {self.beep_counter} times. Shutting down all machines."
        )
        self.message_sender.send(message, severity=Severity.ALERT)

    def turn_off_devices(self):
        os.system("bash shutdown_devices.sh")

    def main(self):
        while True:
            beep = self.beep_classifier.predict_live()
            if int(beep) == 1:
                self.beep_counter += 1
            else:
                self.beep_counter = 0
            print(f"{self.beep_counter} continuous beeps")
            if self.beep_counter >= self.beep_threshold:
                self.handle_fire()


class RemoteFireKeeper(FireKeeper):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.beep_classifier = RemoteBeepClassifier(*args, **kwargs)
