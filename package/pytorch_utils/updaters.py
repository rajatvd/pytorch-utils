

class Updater():
    """A running metric updater"""
    def reset(self):
        """Reset the state of the updater"""
        raise NotImplementedError("reset not implemented")
        
    def update(self, sample):
        """Update the running metric with a new sample and return the updated value"""
        raise NotImplementedError("update not implemented")

class Averager(Updater):
    """A running averager"""
    def reset(self):
        self.count = 0
        self.running_sum = 0
    
    def update(self, sample):
        self.count += 1
        self.running_sum += sample
        return self.running_sum/self.count

class Maxer(Updater):
    """A running updater which takes max of the metric"""
    def reset(self):
        self.running_max = -float('inf')
    
    def update(self, sample):
        self.running_max = max(self.running_max, sample)
        return self.running_max

class Minner(Updater):
    """A running updater which takes min of the metric"""
    def reset(self):
        self.running_min = float('inf')
    
    def update(self, sample):
        self.running_min = min(self.running_min, sample)
        return self.running_min

class Latest(Updater):
    """A running updater which returns the latest sample"""
    def reset(self):
        pass
    def update(self, sample):
        return sample