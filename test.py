import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor




import warnings

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsRegressor
fname = "cropyield_dataset.csv"
df = pd.read_csv("../CropYield_Prediction/dataset/" + fname)

x_train=df[['temperature','humidity','ph','rainfall']]
print(x_train)

y_train=df[['Area','Production']]

r1 = DecisionTreeRegressor()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)
r3 = KNeighborsRegressor()

er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])

clf = MultiOutputRegressor(er).fit(x_train, y_train)

x_test=[[24.88921174,81.97927117,5.005306977,185.9461429]]

print("res=",clf.predict(x_test).tolist()[0])

print(type(clf.predict(x_test)[0]))