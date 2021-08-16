import multiprocessing as mp
from functools import partial


t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_func(t, x, y):
    return t*x*y


if __name__ == '__main__':
    with mp.Pool(2) as p:
        x = 2
        y = 3
        my_func = partial(test_func, x=x, y=y)
        results = p.map(my_func, t)

    print(results)
