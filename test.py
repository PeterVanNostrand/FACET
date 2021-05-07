# from sklearn.ensemble import IsolationForest
# X = [[-1.1], [0.3], [0.5], [100]]
# clf = IsolationForest(random_state=0).fit(X)
# print(clf.predict([[0.1], [0], [90]]))

# import numpy as np
# x = np.array([[1, 2], [3, 4], [5, 6]])
# print(x.shape)

import numpy as np
d = np.array([1, 1, 1, 0, 0, 0])
e = np.array([1, 1, 0, 0, 1, 0])
a = np.array([True, False])
b = np.array([True, True])

print(np.where((d == 1) & (e == 1))[0].shape)

# c = a and b
# print(a)
# print(b)
