# Source on mugegen's github: https://github.com/mugenen/LossyCounting/blob/master/lossy_counting_variant.py

class MugegenLossyCounting(object):
    def __init__(self, epsilon):
        self.N = 0
        self.count = {}
        self.epsilon = epsilon
        self.b_current = 0
    
    def getCount(self, item):
        return self.count[item]
    
    def trim(self):
        for item in list(self.count.keys()):
            if self.count[item] < self.b_current:
                del self.count[item]
        
    def addCount(self, item):
        self.N += 1
        if item in self.count:
            self.count[item] += 1
        else:
            self.count[item] = 1 + self.b_current
        
        if self.N % int(1 / self.epsilon) == 0:
            self.b_current += 1
            self.trim()
    
    def iterateOverThresholdCount(self, threshold_count):
        assert threshold_count > self.epsilon * self.N, "too small threshold"
        
        for item in self.count:
            if self.count[item] >= threshold_count:
                yield (item, self.count[item])
    
    def iterateOverThresholdRate(self, threshold_rate):
        return self.iterateOverThresholdCount(threshold_rate * self.N)
    