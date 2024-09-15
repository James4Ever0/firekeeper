from lib import MessageSender, Severity


def main():
    sender = MessageSender()
    sender.send("test_alert_message", severity=Severity.ALERT)


if __name__ == "__main__":
    main()
