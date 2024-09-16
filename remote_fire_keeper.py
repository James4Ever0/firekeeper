import os

from lib import RemoteFireKeeper

HOST = os.environ.get("AUDIO_RECORDER_SERVER_HOST")


def main():
    fire_keeper = RemoteFireKeeper(HOST)
    fire_keeper.main()


if __name__ == "__main__":
    main()
