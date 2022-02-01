import multiprocessing as mp
import time


def long_function(q, test_string):
    time.sleep(10)
    print(test_string)


if __name__ == '__main__':

    print("hello world")

    maxTime = 5

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=long_function, args=(q, "test"))
    p.start()
    p.join(maxTime)
    if p.is_alive():
        p.terminate()
        print("killing after", maxTime, "second")
    else:
        print("not killed")
