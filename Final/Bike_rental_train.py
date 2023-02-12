import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle #save encoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


df = pd.read_csv('p1_train.csv')
    
X = df.iloc[:, [2,3,4,5,7,9]]
y = df.iloc[:, [11]]

X = X.fillna(0)
Xorg = X


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

# Feature Scaling
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)

y_test = scaler_y.transform(y_test)
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history=model.fit(X_train, y_train, epochs=200, batch_size=20, validation_data=(X_test,y_test))

# Visualisoidaan mallin oppiminen
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.show()

y_pred = scaler_y.inverse_transform(model.predict(X_test))

y_test = scaler_y.inverse_transform(y_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print ('\nann:')
print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'rmse: {rmse}\n')

# tallentaan malli levylle
model.save('bike_rental-ann-model.h5')

    
# save scalers to disk
with open('bike_rental-ann-scaler_x.pickle', 'wb') as f:
    pickle.dump(scaler_x, f)
    
with open('bike_rental-ann-scaler_y.pickle', 'wb') as f:
    pickle.dump(scaler_y, f)
    
    
########################## Prediction code should be seperated to run the code ###########################
##########################################################################################################


import pandas as pd
from tensorflow.keras.models import load_model #load model
import pickle #load encoder

# load model
model = load_model('bike_rental-ann-model.h5')

# load scalers
with open('bike_rental-ann-scaler_x.pickle', 'rb') as f:
    scaler_x = pickle.load(f)

with open('bike_rental-ann-scaler_y.pickle', 'rb') as f:
    scaler_y = pickle.load(f)

# ennusta uudella datalla
Xnew = pd.read_csv('p1_test.csv')
Xnew = Xnew.iloc[:, [2,3,4,5,7,9]]
Xnew_org = Xnew
Xnew = scaler_x.transform(Xnew)
ynew = model.predict(Xnew) 
ynew = scaler_y.inverse_transform(ynew)

# get scaled value back to unscaled
Xnew = scaler_x.inverse_transform(Xnew)

ynew = pd.DataFrame(ynew).reindex()
ynew.columns = ['predicted_count']
df_results = Xnew_org.join(ynew)

# tallennetaan ennusteet csv-tiedostoon
df_results.to_csv('p1-test-ann-with-count.csv', index=False)
