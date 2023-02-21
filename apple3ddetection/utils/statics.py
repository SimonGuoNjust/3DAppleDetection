import time
import queue

class Averager(object):
    def __init__(self):
        self.wholenum = 0
        self.wholevalue = 0.0

    def add(self, value):
        self.wholenum += 1
        self.wholevalue += value

    def average(self):
        return self.wholevalue / self.wholenum if self.wholenum != 0 else 0

    def clean(self):
        self.wholenum = 0
        self.wholevalue = 0.0


class Timer(object):
    def __init__(self, steps, labels=None):
        self.step_list = queue.Queue(steps)
        self.avg_list = [Averager()] * steps
        self.labels = labels

    def start(self):
        self.time0 = time.time()

    def count(self):
        if self.step_list.full():
            for i in range(self.step_list.qsize()):
                self.avg_list[i].add(self.step_list.get())
        self.step_list.put(time.time()-self.time0)
        self.start()

    def show(self):
        show_m = str()
        for label,timer in zip(self.labels,self.avg_list):
            show_m += f"{label}:{timer.average():.4f}s,"
        print(show_m)