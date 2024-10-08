import os

from lib import RemoteMelFireKeeper
import json
recorder_server_host = json.load(open(".secrets.json"))['recorder_server_host']

HOST = os.environ.get("AUDIO_RECORDER_SERVER_HOST", recorder_server_host)


def main():
    fire_keeper = RemoteMelFireKeeper(HOST)
    fire_keeper.main()


if __name__ == "__main__":
    main()
