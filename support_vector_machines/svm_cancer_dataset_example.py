from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# load the cancer dataset
cancer: Bunch = datasets.load_breast_cancer()

# display the dataset feature names
print("Features: ", cancer.feature_names)

# display the dataset target names
print("Labels: ", cancer.target_names)

# see the Bunch object attributes
print(cancer.__dir__())

# print data(feature)shape, returns a tuple
print(cancer.data.shape)

# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

# create a svm Classifier
clf = svm.SVC(kernel='linear')

# train the model using the training sets
clf.fit(x_train, y_train)

# predict the response for test dataset
y_pred = clf.predict(x_test)

# model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))


