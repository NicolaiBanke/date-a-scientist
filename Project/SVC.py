import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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


min_max_scaler = preprocessing.MinMaxScaler()

female_abrahamic_scaled = min_max_scaler.fit_transform(female_abrahamic[["abrahamic", "drinks", "smokes", "drugs"]].values)
female_eastern_scaled = min_max_scaler.fit_transform(female_eastern[["eastern", "drinks", "smokes", "drugs"]].values)
female_non_religious_scaled = min_max_scaler.fit_transform(female_non_religious[["non_religious", "drinks", "smokes", "drugs"]].values)

female_abrahamic_labels = female_abrahamic.orientation.values
female_eastern_labels = female_eastern.orientation.values
female_non_religious_labels = female_non_religious.orientation.values

data = [(female_abrahamic_scaled, female_abrahamic_labels), (female_eastern_scaled, female_eastern_labels), (female_non_religious_scaled, female_non_religious_labels)]
accuracies = [[] for datum in data]
parameters = []

#|-------- Time for loop ----------|
start = time.time()
for i in range(len(data)):
	accuracy = 0
	for c in range(1, 100):
		for g in range(1, 10):
			training_data, validation_data, training_labels, validation_labels = train_test_split(data[i][0], data[i][1], train_size = 0.8, test_size = 0.2, random_state = 100)
			classifier = SVC(gamma=g, C=c)
        	classifier.fit(training_data, training_labels)
        	accuracies.append(classifier.score(validation_data, validation_labels))
        	if accuracy < classifier.score(validation_data, validation_labels):
  	      		accuracy = classifier.score(validation_data, validation_labels)
  	      		(C_max, gamma_max) = (c, g)
	parameters.append((C_max, gamma_max))

end = time.time()
#|-------- Time for loop ----------|
print("Execution time: {0} seconds".format(end - start))

"""start = time.time()
for g in range(1, 10000):
	classifier = SVC(gamma=g, C=1)
	classifier.fit(training_data, training_labels)
	accuracies.append(classifier.score(validation_data, validation_labels))
	if acc < classifier.score(validation_data, validation_labels):
		acc = classifier.score(validation_data, validation_labels)
		gamma_max = g
end = time.time()"""

classifiers = []
scores = []

for i in range(len(data)):
	training_data, validation_data, training_labels, validation_labels = train_test_split(data[i][0], data[i][1], train_size = 0.8, test_size = 0.2, random_state = 100)
	classifier = SVC(kernel="rbf", C=parameters[i][0], gamma=parameters[i][1])
	classifier.fit(training_data, training_labels)
	classifiers.append(classifier)
	scores.append(classifier.score(validation_data, validation_labels))

for i in range(3):
	print("Score for {0}: {1}".format(religions[i], scores[i]))
	print("parameters: {0}".format(parameters[i]))