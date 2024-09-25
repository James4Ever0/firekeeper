import os

from lib import RemoteFireKeeper

HOST = os.environ.get("AUDIO_RECORDER_SERVER_HOST", "192.168.5.104")


def main():
    fire_keeper = RemoteFireKeeper(HOST)
    fire_keeper.main()


if __name__ == "__main__":
    main()
