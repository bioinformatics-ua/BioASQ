import threading

class TestThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.a = 2

    def run(self):
        for i in range(1000000000):
            self.a += (i*20)


th = [TestThread() for _ in range(10)]

for t in th:
    t.start()

for t in th:
    t.join()
