from miio import DeviceFactory

dev = DeviceFactory.create("<ip address>", "<token>")
dev.status()
