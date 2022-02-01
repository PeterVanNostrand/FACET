from dataset import load_data
from sklearn.model_selection import train_test_split

x, y = load_data("thyroid", normalize=True)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

print(xtest[0])
