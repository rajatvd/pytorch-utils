

def averager():
    """A running averager"""
    s = yield
    n = 1
    while True:
        s += yield s/n
        n += 1

def maxxer():
    """A running updater which takes max of the metric"""
    running_max = yield
    while True:
        x = yield running_max
        running_max = max(running_max, x)

def maxxer():
    """A running updater which takes min of the metric"""
    running_min = yield
    while True:
        x = yield running_min
        running_min = min(running_min, x)

def latest():
    """A running updater which returns the latest sample"""
    a = yield
    while True:
        a = yield a