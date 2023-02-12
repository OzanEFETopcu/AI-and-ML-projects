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
