from sklearn import svm
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# total sample size
sample = 10000

# generate male[0] and female[1] datasets
# Height[cm], Weight[kg], Shoe size[UK], gender
male_dataset = [
    (random.randrange(145, 190), random.randrange(45, 80), random.randrange(7, 9), 0)
    for _ in range(sample)
]

female_dataset = [
    (random.randrange(145, 170), random.randrange(30, 70), random.randrange(3, 8), 1)
    for _ in range(sample)
]

# collect all male and female data
total_dataset = male_dataset + female_dataset

# parse the features
parsed_dataset_features = [
    [feature[0], feature[1], feature[2]]
    for feature in total_dataset
]

# parse the targets
parsed_dataset_targets = [
    target[3]
    for target in total_dataset
]

# split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    parsed_dataset_features,
    parsed_dataset_targets,
    test_size=0.3,
    random_state=109
)

# create support vector machine - defaults to rbf kernel
clf = svm.SVC(kernel='linear')

# train the model
clf.fit(x_train, y_train)

# Predict if male or female based on test dataset
p = clf.predict([[152, 68, 7]])

print("Prediction: ", p)
