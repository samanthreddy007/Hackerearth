

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMRegressor

pati_mon_rev_test=pd.read_csv("D:/hackerearth/proteas/patient_monthwise_revenue_test.csv")
pati_mon_rev_train=pd.read_csv("D:/hackerearth/proteas/patient_monthwise_revenue_train.csv")
physio_diag_train=pd.read_csv("D:/hackerearth/proteas/physio_diagnosis_train.csv")
physio_diag_test=pd.read_csv("D:/hackerearth/proteas/physio_diagnosis_test.csv")
patient_train_classify=pd.read_csv("D:/hackerearth/proteas/patient_train_classified.csv")
physio_appts_train=pd.read_csv("D:/hackerearth/proteas/physio_appts_train.csv")
Submission=pd.read_csv("D:/hackerearth/proteas/Submission.csv")

s=pati_mon_rev_train['service_id'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train["service_id"].isin(bad), "service_id"] = 999
pati_mon_rev_test.loc[pati_mon_rev_test["service_id"].isin(bad), "service_id"] = 999

s=pati_mon_rev_train['city'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train["city"].isin(bad), "city"] = 'other'
pati_mon_rev_test.loc[pati_mon_rev_test["city"].isin(bad), "city"] = 'other'


y=pati_mon_rev_train['revenue']
no_of_visits_train=pati_mon_rev_train['visits_count']

pati_mon_rev_train.drop(['revenue','visits_count','Unnamed: 20','Unnamed: 21'],axis=1,inplace=True)
pati_mon_rev_test.drop(['Unnamed: 18','Unnamed: 19'],axis=1,inplace=True)

train_len=pati_mon_rev_train.shape[0]
test_len=pati_mon_rev_test.shape[0]
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['city']=le.fit_transform(full['city'])
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)

le=preprocessing.LabelEncoder()
pati_mon_rev_train['ref_type']=le.fit_transform(pati_mon_rev_train['ref_type'])
pati_mon_rev_test['ref_type']=le.fit_transform(pati_mon_rev_test['ref_type'])

s=pati_mon_rev_train['ref_name'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train["ref_name"].isin(bad), "ref_name"] = 'other'
pati_mon_rev_test.loc[pati_mon_rev_test["ref_name"].isin(bad), "ref_name"] = 'other'

le=preprocessing.LabelEncoder()
full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['ref_name']=le.fit_transform(full['ref_name'])
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)

s=pati_mon_rev_train['ref_source'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train["ref_source"].isin(bad), "ref_source"] = 'Clinic'
pati_mon_rev_test.loc[pati_mon_rev_test["ref_source"].isin(bad), "ref_source"] = 'Clinic'

le=preprocessing.LabelEncoder()
full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['ref_source']=le.fit_transform(full['ref_source'])
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)

s=pati_mon_rev_train['service_name'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train['service_name'].isin(bad), 'service_name'] = 'other'
pati_mon_rev_test.loc[pati_mon_rev_test['service_name'].isin(bad), 'service_name'] = 'other'

le=preprocessing.LabelEncoder()
full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['service_name'] = full['service_name'].factorize()[0]
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)

pati_mon_rev_train.drop(['FVS'],axis=1,inplace=True)
pati_mon_rev_test.drop(['FVS'],axis=1,inplace=True)


full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['gender'] = full['gender'].factorize()[0]
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)


le=preprocessing.LabelEncoder()
full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['brand'] = full['brand'].factorize()[0]
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)


s=pati_mon_rev_train['diagnosis'].value_counts()
bad = s.index[s < 100]
pati_mon_rev_train.loc[pati_mon_rev_train['diagnosis'].isin(bad), 'diagnosis'] = 'other'
pati_mon_rev_test.loc[pati_mon_rev_test['diagnosis'].isin(bad), 'diagnosis'] = 'other'

full=pd.concat([pati_mon_rev_train,pati_mon_rev_test])
full.reset_index(inplace=True, drop=True) 
full['diagnosis'] = full['diagnosis'].factorize()[0]
pati_mon_rev_train=full[:train_len]
pati_mon_rev_test=full[train_len:].reset_index(drop=True)


#pati_mon_rev_train['FVD'] = pd.to_datetime(pati_mon_rev_train['FVD'])
pati_mon_rev_train['FVM_new'] = '1/'+pati_mon_rev_train['FVM'].astype(str)
pati_mon_rev_train['FVM_new'] = pd.to_datetime(pati_mon_rev_train['FVM_new'],dayfirst=True)
pati_mon_rev_train['visit_new'] = '1/'+pati_mon_rev_train['visit_month_year'].astype(str)
pati_mon_rev_train['visit_new'] = pd.to_datetime(pati_mon_rev_train['visit_new'],dayfirst=True)
pati_mon_rev_train['nb_months'] = (pati_mon_rev_train.visit_new - pati_mon_rev_train.FVM_new)/ np.timedelta64(1, 'M')
pati_mon_rev_train['LVD_new'] = pd.to_datetime(pati_mon_rev_train['LVD'],dayfirst=True)
pati_mon_rev_train['LVD_new'] = pati_mon_rev_train['LVD_new'].map(lambda t: t.replace(day=1))
pati_mon_rev_train['mon_to_complete'] = (pati_mon_rev_train.LVD_new - pati_mon_rev_train.visit_new)/ np.timedelta64(1, 'M')

pati_mon_rev_train['visit_month']=pati_mon_rev_train['visit_new'].dt.month
pati_mon_rev_train['visit_year']=pati_mon_rev_train['visit_new'].dt.year
pati_mon_rev_train['visit_year'] =pati_mon_rev_train['visit_year'].factorize()[0]


new = pati_mon_rev_train.drop(['patient_id','visit_month_year','FVD','FVM','LVD','avg_nps','visit_new','FVM_new','LVD_new'], axis=1)

def cat(x):
    if x <1000:
        return "Low"
    elif 999 <x<5000:
        return "Med"
    elif 4999 <x<10000:
        return "High-Med"
    elif x>9999:
        return "High"
    
y_new = y.apply(lambda x: cat(x))
y_new_2 = y_new.factorize()[0]


unique, counts = np.unique(y_new_2, return_counts=True)

print (np.asarray((unique, counts)).T)
new=new.replace('\\N',0)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=99)
param_grid = {
                 'n_estimators': [5, 10, 15, 20],
                 'max_depth': [2, 5, 7, 9]
             }
from sklearn.grid_search import GridSearchCV

#grid_clf = GridSearchCV(rf_model, param_grid, cv=5)
#grid_clf.fit(new, y)
#grid_clf. best_params_
#grid_clf.grid_scores_
rf_model.fit(new,y)

pati_mon_rev_test['FVM_new'] = '1/'+pati_mon_rev_test['FVM'].astype(str)
pati_mon_rev_test['FVM_new'] = pd.to_datetime(pati_mon_rev_test['FVM_new'],dayfirst=True)
pati_mon_rev_test['visit_new'] = '1/'+pati_mon_rev_test['visit_month_year'].astype(str)
pati_mon_rev_test['visit_new'] = pd.to_datetime(pati_mon_rev_test['visit_new'],dayfirst=True)
pati_mon_rev_test['nb_months'] = (pati_mon_rev_test.visit_new - pati_mon_rev_test.FVM_new)/ np.timedelta64(1, 'M')
pati_mon_rev_test['LVD_new'] = pd.to_datetime(pati_mon_rev_test['LVD'],dayfirst=True)
pati_mon_rev_test['LVD_new'] = pati_mon_rev_test['LVD_new'].map(lambda t: t.replace(day=1))
pati_mon_rev_test['mon_to_complete'] = (pati_mon_rev_test.LVD_new - pati_mon_rev_test.visit_new)/ np.timedelta64(1, 'M')

pati_mon_rev_test['visit_month']=pati_mon_rev_test['visit_new'].dt.month
pati_mon_rev_test['visit_year']=pati_mon_rev_test['visit_new'].dt.year
pati_mon_rev_test['visit_year'] =pati_mon_rev_test['visit_year'].factorize()[0]

new_x = pati_mon_rev_test.drop(['patient_id','visit_month_year','FVD','FVM','LVD','avg_nps','visit_new','FVM_new','LVD_new'], axis=1)
new_x=new_x.replace('\\N',0)
predictions=rf_model.predict(new_x)
#predictions=grid_clf.predict(new_x)

pati_mon_rev_test['revenue']=predictions
pred=pati_mon_rev_test.groupby('patient_id')['revenue'].agg('sum')


pred_new = pred.apply(lambda x: cat(x))
s=pred_new.to_frame()
s['index1'] = s.index
s.rename(columns={'index1': 'PID'}, inplace=True)
f=pd.merge(Submission,s,on='PID')



f.drop(['Bucket'],axis=1,inplace=True)

f.rename(columns={'revenue': 'Bucket'}, inplace=True)

f.to_csv('D:/hackerearth/sample_submission_39.csv',index=False)