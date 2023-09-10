import pandas as pd
from sklearn.datasets import load_iris

# Set display options
pd.set_option('display.max_columns', None)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = iris.target
df['class name'] = iris.target_names[iris['target']]
print(df.head())


# split test/training set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

# train the model
from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier()
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x_train, y_train)

# exercise model with test data
print(model.score(x_test, y_test))

# predict type with data
print(model.predict([[5.6, 4.4, 1.2, 0.4]]))

