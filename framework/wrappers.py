import time

def ftime(func):
    def wrap(*args, **kwargs):
        print("\r[INFO] %s : Working" % (func.__name__), end = '\r', flush = True)
        start = time.time()
        solution = tmp if (tmp:=func(*args, **kwargs)) is not None else [None]
        runtime = round(time.time()-start, 2)
        print("[INFO] <%ss> %s: Done" % (runtime, (func.__name__)))
        return (*solution, round(time.time()-start, 2))
    return wrap