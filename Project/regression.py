import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from religion import eastern_mappings, abrahamic_mappings, non_religious_mappings
from sklearn import preprocessing

religion_mappings_to_numbers = {"none": 0, "little": 1, "some": 2, "somewhat": 3, "very": 4}

df = pd.read_csv("profiles.csv")
male = df[["drugs", "drinks", "smokes", "religion", "orientation"]][df.sex == "m"].dropna()
female = df[["drugs", "drinks", "smokes", "religion", "orientation"]][df.sex == "f"]

drugs_mappings = {"never": 0, "sometimes": 1, "often": 2}
drinks_mappings = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mappings = {"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4}


vices = ["drugs", "drinks", "smokes"]
mappings = [drugs_mappings, drinks_mappings, smokes_mappings]

for i in range(3):
	female[vices[i]] = female[vices[i]].dropna().map(mappings[i])


female["abrahamic"] = female.religion.map(abrahamic_mappings).map(religion_mappings_to_numbers)
female["eastern"] = female.religion.map(eastern_mappings).map(religion_mappings_to_numbers)
female["non_religious"] = female.religion.map(non_religious_mappings).map(religion_mappings_to_numbers)

female_abrahamic = female[["abrahamic", "drinks", "smokes", "drugs"]].dropna()
female_eastern = female[["eastern", "drinks", "smokes", "drugs"]].dropna()
female_non_religious = female[["non_religious", "drinks", "smokes", "drugs"]].dropna()


min_max_scaler = preprocessing.MinMaxScaler()

y_abrahamic_scaled = min_max_scaler.fit_transform(female_abrahamic.abrahamic.values.reshape(-1, 1))
x_abrahamic_scaled = min_max_scaler.fit_transform(female_abrahamic[["drinks", "smokes", "drugs"]].values)

y_eastern_scaled = min_max_scaler.fit_transform(female_eastern.eastern.values.reshape(-1, 1))
x_eastern_scaled = min_max_scaler.fit_transform(female_eastern[["drinks", "smokes", "drugs"]].values)

y_non_religious_scaled = min_max_scaler.fit_transform(female_non_religious.non_religious.values.reshape(-1, 1))
x_non_religious_scaled = min_max_scaler.fit_transform(female_non_religious[["drinks", "smokes", "drugs"]].values)

data = [(x_abrahamic_scaled, y_abrahamic_scaled), (x_eastern_scaled, y_eastern_scaled), (x_non_religious_scaled, y_non_religious_scaled)]

model = LinearRegression()

religions = ["Abrahamic", "Eastern", "Non-Religious"]

for i in range(3):
	x_train, x_test, y_train, y_test = train_test_split(data[i][0], data[i][1], train_size = 0.8, test_size = 0.2, random_state = 100)

	model.fit(x_train, y_train)
	print("|-------------------------------{0}----------------------------------|".format(religions[i]))
	print("coef_ = {0}".format(model.coef_))
	print("intercept_ = {0}".format(model.intercept_))
	print("score train: {0}".format(model.score(x_train, y_train)))
	print("score test: {0}".format(model.score(x_test, y_test)))

print("|-----------------------------------------------------------------|")