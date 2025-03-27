import os
with open("/Users/igorbulgakov/Documents/vib_bot/env.log", "a") as f:
    for key, value in os.environ.items():
        f.write(f"{key}={value}\n")
#!/usr/bin/env python3
import time
with open("/Users/igorbulgakov/Documents/vib_bot/test.log", "a") as f:
    f.write("Test script started at " + time.ctime() + "\n")
while True:
    with open("/Users/igorbulgakov/Documents/vib_bot/test.log", "a") as f:
        f.write("Heartbeat at " + time.ctime() + "\n")
    time.sleep(60)