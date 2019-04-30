import time

class Stopwatch:
    def  __init__(self, time_unit = 's', decimals = 2):
        self.start_time = None
        self.stop_time = None
        self.time_unit = time_unit
        self.decimals = decimals
        self.registred_times = {}

    def start(self, name = None):
        if self.start_time == None:
            self.start_time = time.time()
        else:
            raise Exception("Stopwatch start time is already set")
    
    def stop(self):
        if self.start_time == None:
            raise Exception("Stopwatch start time is not set")
        if self.stop_time != None:
            raise Exception("Stopwatch stop time is already set")
        self.stop_time = time.time()
        return self.calculate_time()

    def clear(self):
        self.start_time = None
        self.stop_time = None

    def calculate_time(self):
        result = self.truncate(self.stop_time - self.start_time, self.decimals)
        if self.time_unit == 's':
            return result
        if self.time_unit == 'ms':
            return result * 1000
        if self.time_unit == 'm':
            return result / 60
        if self.time_unit == 'h':
            return result / 3600

    def truncate(self, n, decimals):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
