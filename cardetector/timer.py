import time

class Timer(object):
    def __init__(self):
        self._duration = 0

    def start(self):
        self._duration = time.time()

    def stop(self):
        self._duration = time.time() - self._duration

    def print(self, msg):
        print(msg.format(self._duration))
