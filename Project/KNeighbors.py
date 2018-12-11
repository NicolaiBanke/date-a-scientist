import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
import time

df = pd.read_csv("profiles.csv")
male = df[["drugs", "drinks", "religion", "smokes", "orientation"]][df.sex == "m"]
female = df[["drugs", "drinks", "religion", "smokes", "orientation"]][df.sex == "f"]

vices = ["drugs", "drinks", "smokes"]


"""for vice in vices:
	print(male[vice].value_counts())
	print("|-----------------------------------------|")
"""
drugs_mappings = {"never": 0, "sometimes": 1, "often": 2}
drinks_mappings = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mappings = {"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4}
eastern_mappings = {
	"buddhism but not too serious about it": "little" ,
	"buddhism and laughing about it": "none" ,
	"buddhism": "some" ,
	"buddhism and somewhat serious about it": "somewhat",
	"hinduism but not too serious about it": "little",
	"hinduism": "some",
	"buddhism and very serious about it": "very",
	"hinduism and somewhat serious about it": "somewhat",
	"hinduism and laughing about it": "none",
	"hinduism and very serious about it": "very"
}

abrahamic_mappings = {
	"catholicism but not too serious about it": "little" ,
	"christianity": "some" ,
	"christianity but not too serious about it": "little" ,
	"judaism but not too serious about it": "little" ,
	"catholicism": "some",
	"christianity and somewhat serious about it": "somewhat" ,
	"catholicism and laughing about it": "none" ,
	"judaism and laughing about it": "none" ,
	"judaism": "some",
	"christianity and very serious about it": "very" ,
	"catholicism and somewhat serious about it": "somewhat",
	"christianity and laughing about it": "none" ,
	"judaism and somewhat serious about it": "somewhat" ,
	"catholicism and very serious about it": "very" ,
	"islam": "some",
	"islam but not too serious about it": "little" ,
	"judaism and very serious about it": "very",
	"islam and somewhat serious about it": "somewhat" ,
	"islam and laughing about it": "none" ,
	"islam and very serious about it": "very"
}

non_religious_mappings = {
	"agnosticism": "some",
	"other": "some",
	"agnosticism but not too serious about it": "little" ,
	"agnosticism and laughing about it": "none" ,
	"atheism": "some",
	"other and laughing about it": "none" ,
	"atheism and laughing about it": "none" ,
	"other but not too serious about it": "little" ,
	"atheism but not too serious about it": "little" ,
	"atheism and somewhat serious about it": "somewhat" ,
	"other and somewhat serious about it": "somewhat" ,
	"agnosticism and somewhat serious about it": "somewhat" ,
	"atheism and very serious about it": "very" ,
	"other and very serious about it": "very" ,
	"agnosticism and very serious about it": "very"
}


mappings = [drugs_mappings, drinks_mappings, smokes_mappings]

for i in range(3):
	female[vices[i]] = female[vices[i]].dropna().map(mappings[i])


religion_mappings_to_numbers = {"none": 0, "little": 1, "some": 2, "somewhat": 3, "very": 4}

religions = ["Abrahamic", "Eastern", "Non-Religious"]

female["abrahamic"] = female.religion.map(abrahamic_mappings).map(religion_mappings_to_numbers)
female["eastern"] = female.religion.map(eastern_mappings).map(religion_mappings_to_numbers)
female["non_religious"] = female.religion.map(non_religious_mappings).map(religion_mappings_to_numbers)

female_abrahamic = female[["orientation", "abrahamic", "drinks", "smokes", "drugs"]].dropna()
female_eastern = female[["orientation", "eastern", "drinks", "smokes", "drugs"]].dropna()
female_non_religious = female[["orientation", "non_religious", "drinks", "smokes", "drugs"]].dropna()

female_religions = [female_abrahamic, female_eastern, female_non_religious]

"""f, ax = plt.subplots(1, 3)
for i in range(len(female_religions)):
	ax[i].pie(female_religions[i].orientation.value_counts(), labels=female_religions[i].orientation.value_counts().index, autopct='%0.1f%%', shadow=True)
	ax[i].set_title("{0}".format(religions[i]))
	ax[i].axis('equal')

plt.show()"""

min_max_scaler = preprocessing.MinMaxScaler()

female_abrahamic_scaled = min_max_scaler.fit_transform(female_abrahamic[["abrahamic", "drinks", "smokes", "drugs"]].values)
female_eastern_scaled = min_max_scaler.fit_transform(female_eastern[["eastern", "drinks", "smokes", "drugs"]].values)
female_non_religious_scaled = min_max_scaler.fit_transform(female_non_religious[["non_religious", "drinks", "smokes", "drugs"]].values)

female_abrahamic_labels = female_abrahamic.orientation.values
female_eastern_labels = female_eastern.orientation.values
female_non_religious_labels = female_non_religious.orientation.values

data = [(female_abrahamic_scaled, female_abrahamic_labels), (female_eastern_scaled, female_eastern_labels), (female_non_religious_scaled, female_non_religious_labels)]
accuracies = [[] for datum in data]
k_maxs = []

start = time.time()
for i in range(len(data)):
	accuracy = 0
	for k in range(1, 100):
		training_data, validation_data, training_labels, validation_labels = train_test_split(data[i][0], data[i][1], train_size = 0.8, test_size = 0.2, random_state = 100)
		classifier = KNeighborsClassifier(n_neighbors=k)
		classifier.fit(training_data, training_labels)
		accuracies[i].append(classifier.score(validation_data, validation_labels))
		if accuracy < classifier.score(validation_data, validation_labels):
			accuracy = classifier.score(validation_data, validation_labels)
			k_max = k
	k_maxs.append(k_max)

end = time.time()
print("Execution time: {0} seconds".format(end - start))
print("k_maxs = {0}".format(k_maxs))

f, ax = plt.subplots(1, 3)
k_list = range(1, 100)
for i in range(len(data)):
	ax[i].plot(k_list, accuracies[i])
	ax[i].set_xlabel("k")
	ax[i].set_ylabel("Validation Accuracy")
	ax[i].set_title("Orientation Classifier Accuracy for " + "{0}".format(religions[i]))

plt.show()

scores = []
classifiers = []

for i in range(len(data)):
	training_data, validation_data, training_labels, validation_labels = train_test_split(data[i][0], data[i][1], train_size = 0.8, test_size = 0.2, random_state = 100)
	classifier = KNeighborsClassifier(n_neighbors=k_maxs[i])
	classifier.fit(training_data, training_labels)
	classifiers.append(classifier)
	scores.append(classifier.score(validation_data, validation_labels))

for i in range(3):
	print("Score for {0}: {1}".format(religions[i], scores[i]))

"""print("Score: {0}".format(classifier.score(validation_data, validation_labels)))

print("Accuracy: {0}".format(accuracy_score(validation_labels, guesses)))
print("Recall: {0}".format(recall_score(validation_labels, guesses, average=None)))
print("Precision: {0}".format(precision_score(validation_labels, guesses, average=None)))
print("F1: {0}".format(f1_score(validation_labels, guesses, average=None)))"""