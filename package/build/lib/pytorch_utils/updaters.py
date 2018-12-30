def momentum_averager(momentum=0.9):
    """First order running averager with momentum.
    y_n = momentum*y_{n-1} + (1-momentum)*x_n"""
    def avg():
        y = yield
        while True:
            x = yield y
            y = momentum*y + (1-momentum)*x

    return avg

# %%

def averager():
    """A running averager"""
    s = yield
    n = 1
    while True:
        s += yield s/n
        n += 1

# # %%
# # testing momentum averager
# g = momentum_averager(0.9)()
# next(g)
#
# from pylab import *
# t = linspace(0,600,300)
# x = 0.0*sin(t)+0.1*randn(300)+(1-exp(-0.1*t))
# x.shape
# y = []
# for x_ in x:
#     y.append(g.send(x_))
#
# avg = averager()
# next(avg)
# y_avg=[]
# for x_ in x:
#     y_avg.append(avg.send(x_))
#
# plot(x)
# plot(y)
# plot(y_avg)
# show()
# # %%

def maxxer():
    """A running updater which takes max of the metric"""
    running_max = yield
    while True:
        x = yield running_max
        running_max = max(running_max, x)

def minner():
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