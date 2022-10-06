from sklearn.preprocessing import StandardScaler 
import pandas as pa
import numpy as np
names = ['height','weight'] 
dataframe =pa.read_csv(r"C:\Users\riyam\OneDrive - vit.ac.in\SEMESTER 5\AI\LAB\EXPERIMENT 1\data.csv", names=names) 
dataframe.head()
array = dataframe.values 

X = array[:,:-1] 
Y = array[:,-1] 
scaler = StandardScaler()
rescaledX = scaler.fit_transform(X) 

np.set_printoptions(precision=3) 
print(rescaledX[0:10,:])
