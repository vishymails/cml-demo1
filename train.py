from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# Read the data

X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# fit the model 

depth = 2
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)

print(f"Accuracy: {acc}")

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc} \n")

with open("metrics.txt", "r") as f:
    print(f.readlines())

# plot it 

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)

plt.savefig("plot.png")

