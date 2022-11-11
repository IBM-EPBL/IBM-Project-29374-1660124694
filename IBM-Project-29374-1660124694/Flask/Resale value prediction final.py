import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle
df =  pd.read_csv("Data/autos.csv",header=0,sep= ',',encoding='Latin1',)
print(df.seller.value_counts()) 
df[df.seller != 'gewerblich'] 
df=df.drop('seller',1)
print(df.offerType.value_counts())
df[df.offerType != 'Gesuch'] 
df=df.drop('offerType',1)

print(df.shape) 
df = df[(df.powerPS > 50) & (df.powerPS <900)] 
print(df.shape) 

print(df.shape)
df = df[(df .yearOfRegistration >= 1950) & (df.yearOfRegistration < 2017)] 
print(df.shape)
df.drop(['name', 'abtest', 'dateCrowled','nrofpictures', 'lastseen','postalCode', 'duteCreated']. axis-'columnis',inplace-True)
new_df = df.copy()
new_df = new_df.drop_duplicates(['price', 'vehicleType', 'yearofRegistration',
'gearbox', 'powerps', 'model', 'kilometer', 'monthofRegistration', 'fuel Type' ,'notRepairedDamage'])

new_df.gearbox.replace(('manuell', 'automatik'), ('manual', 'automatic'), inplace=True)
new_df. fuelType.replace(('benzin', 'andere', 'elektro'), ('petrol', 'others', 'electric'), inplace=True) 
new_df.vehicleType.replace(('kleinwagen', 'cabrio', 'kombi', 'andere'),
('small car', 'convertible', 'combination', 'others'), inplace=True)
new_df.notRepairedDamage.replace(('ja', 'nein'), ('Yes', 'No'), inplace=True)

new_df = new_df [(new_df.price >= 100) & (new_df.price <= 150000)]
new_df('notRepairedDamage').fillna (value='not-declared', inplace=True)
new_df('fuel Type').fillna(value='not-declared', inplace=True)
new_df('gearbox').fillna (value='not-declared', inplace=True) 
new_df('vehicleType').fillna(value='not-declared', inplace=True)
new_df('model').fillna(value='not-declared', inplace=True)
new_df.to_csv('autos_preprocessed.csv') 
labels = ('gearbox', 'notRepairedDamage', 'model', 'brand', 'fuel Type', 'vehicleType')
mapper = {}
for i in labels:
 mapper[i] = LabelEncoder() 
mapper[i].fit(new_df[i])
tr = mapper[i].transform(new_df[i])
np.save(str('classes'+i+'.npy'), mapper[i].classes) 
print(i,":", mapper[i]) 
new_df.loc[:, i + '_labels'] = pd. Series(tr, index=new_df.index),
labeled = new_df[ ['price', 'yearOfRegistration', 'powerPS', 'kilometer', 'monthOfRegistration' ] + [x+"_labels" for x in labels ]]
print(labeled.columns)
Y = labeled.iloc[:,0].values
X = labeled.iloc[:,1:].values
Y = Y.reshape(-1,1)

from sklearn.model_selection import cross_val_score, train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 3)

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score 
regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=34),

regressor.fit(X_train, np.ravel(Y_train,order='C'))
y_pred = regressor.predict(X_test);
print(r2_score(y_true, y_pred))
filename = 'resale_model.sav'
pickle.dump(regressor, open(filename, 'wb'))