import pandas as pa
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

names = ['height','weight'] 
dataframe =pa.read_csv(r"C:\Users\riyam\OneDrive - vit.ac.in\SEMESTER 5\AI\LAB\EXPERIMENT 1\data.csv", names=names) 
dataframe.head()
array = dataframe.values 

X = array[:,:-1] 
Y = array[:,-1] 
min= MinMaxScaler(feature_range=(-1, 1)) 
scal= min.fit(X)
rescaledX=scal.transform(X) 

np.set_printoptions(precision=3) 
print("Rescaled X:")
print(rescaledX[0:10,:])
