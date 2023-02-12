import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df=pd.read_csv('startup.csv')


# Seperate the dataset
x0=df.iloc[:,0:3]
y=df.loc[:,"Profit"]


# Split your dataset for training and testing
x0_train, x0_test, y_train, y_test = train_test_split(x0, y, test_size = 1/5, random_state = 0)



# Train the model
model0 = LinearRegression()
model0.fit(x0_train, y_train)


# Create a prediction
y0_pred = model0.predict(x0_test)


# Calculating the regression metrics
mae0 = mean_absolute_error(y_test, y0_pred)
mse0 = mean_squared_error(y_test, y0_pred)
rmse0 = np.sqrt(mse0)
r2 = r2_score(y_test, y0_pred)
print ('\nTestdata metrics:')
print (f'mae: {mae0}')
print (f'mse: {mse0}')
print (f'rmse: {rmse0}')
print(f'R2: {r2}') 



new_com_sp = pd.DataFrame([[165349.20 , 136897.80 , 471784.10]], columns=['R&D Spend','Administration','Marketing Spend'])
print (f'Expected company profit with the entered spendings: {np.round(model0.predict(new_com_sp),2)}\n' )