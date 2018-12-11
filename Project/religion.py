import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

#Create your df here:
df = pd.read_csv("profiles.csv")

#print(df[["orientation", "religion"]].head(10))

"""print(df.orientation[df.sex == "f"].value_counts().values)
print(df.orientation[df.sex == "m"].value_counts())"""
#print(df.religion.value_counts())

"""print(df[df["religion"].astype(str).str.contains("atheism")].religion)
print(df[df["religion"].astype(str).str.contains("other")].religion)
print(df[df["religion"].astype(str).str.contains("agnostic")].religion)

print(df[df["religion"].astype(str).str.contains("christianity")].religion)
print(df[df["religion"].astype(str).str.contains("islam")].religion)
print(df[df["religion"].astype(str).str.contains("catholicism")].religion)
print(df[df["religion"].astype(str).str.contains("judaism")].religion)
"""
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

min_max_scaler = MinMaxScaler()

df["eastern"] = df.religion.map(eastern_mappings)
df["abrahamic"] = df.religion.map(abrahamic_mappings)
df["non_religious"] = df.religion.map(non_religious_mappings)

column_order = ['none', 'little', 'some', 'somewhat', 'very']

male_eastern = df.eastern[df.sex == "m"].dropna()
male_abrahamic = df.abrahamic[df.sex == "m"].dropna()
male_non_religious = df.non_religious[df.sex == "m"].dropna()

female_eastern = df.eastern[df.sex == "f"].dropna()
female_abrahamic = df.abrahamic[df.sex == "f"].dropna()
female_non_religious = df.non_religious[df.sex == "f"].dropna()

if __name__ == '__main__':
	male_eastern = male_eastern.value_counts()
	male_abrahamic = male_abrahamic.value_counts()
	male_non_religious = male_non_religious.value_counts()

	female_eastern = female_eastern.value_counts()
	female_abrahamic = female_abrahamic.value_counts()
	female_non_religious = female_non_religious.value_counts()

	#Update dataframes to have columns in same order
	male_eastern = male_eastern[column_order]
	male_abrahamic = male_abrahamic[column_order]
	male_non_religious = male_non_religious[column_order]

	female_eastern = female_eastern[column_order]
	female_abrahamic = female_abrahamic[column_order]
	female_non_religious = female_non_religious[column_order]

	fig, ax = plt.subplots(2, 3)
	ax[0, 0].bar(male_eastern.index, male_eastern/male_eastern.sum())
 	ax[0, 0].set_ylim([0.0, 0.5])
 	ax[0, 0].set_title('Male Eastern')
	ax[0, 1].bar(male_abrahamic.index, male_abrahamic/male_abrahamic.sum())
	ax[0, 1].set_ylim([0.0, 0.5])
	ax[0, 1].set_title('Male Abrahamic')
 	ax[0, 2].bar(male_non_religious.index, male_non_religious/male_non_religious.sum())
	ax[0, 2].set_ylim([0.0, 0.5])
	ax[0, 2].set_title('Male Non-Religious')

	ax[1, 0].bar(female_eastern.index, female_eastern/female_eastern.sum())
	ax[1, 0].set_ylim([0.0, 0.5])
	ax[1, 0].set_title('Female Eastern')
	ax[1, 1].bar(female_abrahamic.index, female_abrahamic/female_abrahamic.sum())
	ax[1, 1].set_ylim([0.0, 0.5])
	ax[1, 1].set_title('Female Abrahamic')
	ax[1, 2].bar(female_non_religious.index, female_non_religious/female_non_religious.sum())
	ax[1, 2].set_ylim([0.0, 0.5])
	ax[1, 2].set_title('Female Non-Religious')
	plt.show()

	print(male_eastern/male_eastern.sum())