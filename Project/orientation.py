import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from religion import eastern_mappings, abrahamic_mappings, non_religous_mappings

#Create your df here:
df = pd.read_csv("profiles.csv")

female_orientation = df.orientation[df.sex == "f"].value_counts()
male_orientation = df.orientation[df.sex == "m"].value_counts()
#print(male_orientation)
female_orientation = male_orientation[['straight', 'bisexual', 'gay']]
print(female_orientation)
#print(male_orientation)

if __name__ == '__main__':
	f, (ax1, ax2) = plt.subplots(1, 2)
	ax1.pie(male_orientation, labels=male_orientation.index, autopct='%0.1f%%', shadow=True)
	ax1.set_title('Male')
	ax1.axis('equal')
	ax2.pie(female_orientation, labels=female_orientation.index, autopct='%0.1f%%', shadow=True)
	ax2.set_title('Female')
	ax2.axis('equal')
	plt.show()

"""print(df.height[df.sex == "f"].value_counts())
print(df.height[df.sex == "m"].value_counts())"""