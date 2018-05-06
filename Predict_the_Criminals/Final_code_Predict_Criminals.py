

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef


train = pd.read_csv("D:/hackerearth/train.csv")
test = pd.read_csv("D:/hackerearth/test.csv")


# Preprocessing 
id_test = test['PERID'].values
target_train = train['Criminal']
train = train.drop(['PERID','Criminal','VESTR'], axis = 1)
test = test.drop(['PERID','VESTR'], axis = 1)

for col in train.columns :
    
    train[col] = train[col].replace(-1,train[col].mode()[0])



for col in test.columns :
    
    test[col] = test[col].replace(-1,test[col].mode()[0])


#train['GRPHLTIN']=train['GRPHLTIN'].replace((85,98,94,97,85),2)
rand_state = np.random.RandomState(99)
#train['IFATHER']=train['IFATHER'].replace(3,2)
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0)

for each in train.columns.difference(['ANALWT_C']):
    dummies = pd.get_dummies(dataset[each], prefix=each, drop_first=False)
    dataset = pd.concat([dataset, dummies], axis=1)
    dataset=dataset.drop(each,axis=1)
        


train = dataset[:train_objs_num]
test = dataset[train_objs_num:]


print(train.values.shape, test.values.shape)
def my_custom_loss_func(ground_truth, predictions):
         
    return matthews_corrcoef(ground_truth, predictions) 

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
         

    def fit_predict(self, X, y, T):
      
        score = make_scorer(my_custom_loss_func, greater_is_better=True)
        
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_holdout = X.iloc[test_idx]
#                
                if 1:
       
                    pos = pd.Series(y_train == 1)
                    trn_dat = pd.concat([  X_train,   X_train.loc[pos]], axis=0)
                    trn_tgt = pd.concat([y_train, y_train.loc[pos]], axis=0)
       
                    idx = np.arange(len(trn_dat))
                    rand_state.seed(99)
                    rand_state.shuffle(idx)
                    trn_dat = trn_dat.iloc[idx]
                    trn_tgt = trn_tgt.iloc[idx]
                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                
                clf.fit(trn_dat,trn_tgt) 
              
                y_pred = clf.predict_proba(X_holdout)[:,1]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)
 
        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        results_2 = cross_val_score(self.stacker, S_train, y, cv=3, scoring='precision')
        results_3 = cross_val_score(self.stacker, S_train, y, cv=3, scoring='recall')
        results_4=cross_val_score(self.stacker, S_train, y, cv=3, scoring=score)
        print("Stacker score: %.5f" % (results.mean()))
        print("Stacker_recall_score: %.5f" % (results_2.mean()))
        print("Stacker_precision_score: %.5f" % (results_3.mean()))
        print("MCC_score: %.5f" % (results_4.mean()))
        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


        
# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 950
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['seed'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['seed'] = 99



log_params={}
log_params['class_weight']={0:1,1:4.5}
log_params['random_state']=99
lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

log_model = LogisticRegression(**log_params)

        
stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2))        
train=train.drop(
['PRXYDATA_98','PRXYDATA_97', 'PRXYDATA_2', 'PRXRETRY_98', 'PRXRETRY_97', 'PRVHLTIN_98', 
 'PRVHLTIN_97', 'PRVHLTIN_85', 'MEDICARE_98', 'MEDICARE_97', 'MEDICARE_85', 'IRWELMOS_7',
 'IRWELMOS_10', 'IRPRVHLT_2', 'IRPINC3_7', 'IRPINC3_6', 'IROTHHLT_99', 'IROTHHLT_1', 
 'IIOTHHLT_3', 'IIOTHHLT_1', 'IIMEDICR_3', 'IIHHSIZ2_3','IIHHSIZ2_1', 'HLTINNOS_99', 
 'HLTINNOS_98', 'HLTINNOS_97', 'HLTINNOS_94', 'HLTINNOS_2', 'HLTINNOS_1', 'HLNVSOR_98',
 'HLNVSOR_97', 'HLNVSOR_94',
  'HLNVSOR_6', 'HLNVSOR_1', 'HLNVREF_97', 'HLNVREF_94', 
  'HLNVREF_6', 'HLNVREF_1', 'HLNVOFFR_98', 'HLNVOFFR_97',
  'HLNVOFFR_94', 'HLNVOFFR_6', 'HLNVOFFR_1', 'HLNVNEED_98',
  'HLNVNEED_97', 'HLNVNEED_94', 'HLNVNEED_6', 'HLNVNEED_1', 
  'HLNVCOST_99', 'HLNVCOST_98', 'HLNVCOST_97', 'HLNVCOST_94',
  'HLNVCOST_6', 'HLNVCOST_1', 'HLLOSRSN_99', 'HLLOSRSN_98',
  'HLLOSRSN_97', 'HLLOSRSN_94', 'HLLOSRSN_9', 'HLLOSRSN_85', 
  'HLLOSRSN_8', 'HLLOSRSN_7', 'HLLOSRSN_6', 'HLLOSRSN_5',
 'HLLOSRSN_4', 'HLLOSRSN_3', 'HLLOSRSN_2', 'HLLOSRSN_12', 
  'HLLOSRSN_11', 'HLLOSRSN_10', 'HLLOSRSN_1', 'HLCNOTYR_99', 
  'HLCNOTYR_85', 'HLCNOTMO_97', 'HLCNOTMO_85', 'HLCLAST_98', 
  'HLCLAST_97', 'HLCLAST_94', 'HLCLAST_5', 'HLCLAST_4',
 'HLCLAST_3', 'HLCLAST_2', 'HLCLAST_1', 'HLCALLFG_98', 'HLCALLFG_1',
 'HLCALL99_98', 'HLCALL99_1', 'GRPHLTIN_97', 'GRPHLTIN_85', 
  'CHAMPUS_98', 'CHAMPUS_85', 'CELLWRKNG_98', 'CELLWRKNG_97',
  'CELLWRKNG_85', 'CELLNOTCL_98', 'CELLNOTCL_97', 'CELLNOTCL_85'
,  'CAIDCHIP_98', 'CAIDCHIP_97', 'ANYHLTI2_98', 'ANYHLTI2_97', 
'ANYHLTI2_2','ANYHLTI2_1' ],axis=1)

test=test.drop(
['PRXYDATA_98','PRXYDATA_97', 'PRXYDATA_2', 'PRXRETRY_98', 'PRXRETRY_97', 'PRVHLTIN_98', 
 'PRVHLTIN_97', 'PRVHLTIN_85', 'MEDICARE_98', 'MEDICARE_97', 'MEDICARE_85', 'IRWELMOS_7',
 'IRWELMOS_10', 'IRPRVHLT_2', 'IRPINC3_7', 'IRPINC3_6', 'IROTHHLT_99', 'IROTHHLT_1', 
 'IIOTHHLT_3', 'IIOTHHLT_1', 'IIMEDICR_3', 'IIHHSIZ2_3','IIHHSIZ2_1', 'HLTINNOS_99', 
 'HLTINNOS_98', 'HLTINNOS_97', 'HLTINNOS_94', 'HLTINNOS_2', 'HLTINNOS_1', 'HLNVSOR_98',
 'HLNVSOR_97', 'HLNVSOR_94',
  'HLNVSOR_6', 'HLNVSOR_1', 'HLNVREF_97', 'HLNVREF_94', 
  'HLNVREF_6', 'HLNVREF_1', 'HLNVOFFR_98', 'HLNVOFFR_97',
  'HLNVOFFR_94', 'HLNVOFFR_6', 'HLNVOFFR_1', 'HLNVNEED_98',
  'HLNVNEED_97', 'HLNVNEED_94', 'HLNVNEED_6', 'HLNVNEED_1', 
  'HLNVCOST_99', 'HLNVCOST_98', 'HLNVCOST_97', 'HLNVCOST_94',
  'HLNVCOST_6', 'HLNVCOST_1', 'HLLOSRSN_99', 'HLLOSRSN_98',
  'HLLOSRSN_97', 'HLLOSRSN_94', 'HLLOSRSN_9', 'HLLOSRSN_85', 
  'HLLOSRSN_8', 'HLLOSRSN_7', 'HLLOSRSN_6', 'HLLOSRSN_5',
 'HLLOSRSN_4', 'HLLOSRSN_3', 'HLLOSRSN_2', 'HLLOSRSN_12', 
  'HLLOSRSN_11', 'HLLOSRSN_10', 'HLLOSRSN_1', 'HLCNOTYR_99', 
  'HLCNOTYR_85', 'HLCNOTMO_97', 'HLCNOTMO_85', 'HLCLAST_98', 
  'HLCLAST_97', 'HLCLAST_94', 'HLCLAST_5', 'HLCLAST_4',
 'HLCLAST_3', 'HLCLAST_2', 'HLCLAST_1', 'HLCALLFG_98', 'HLCALLFG_1',
 'HLCALL99_98', 'HLCALL99_1', 'GRPHLTIN_97', 'GRPHLTIN_85', 
  'CHAMPUS_98', 'CHAMPUS_85', 'CELLWRKNG_98', 'CELLWRKNG_97',
  'CELLWRKNG_85', 'CELLNOTCL_98', 'CELLNOTCL_97', 'CELLNOTCL_85'
,  'CAIDCHIP_98', 'CAIDCHIP_97', 'ANYHLTI2_98', 'ANYHLTI2_97', 
'ANYHLTI2_2','ANYHLTI2_1'  ],axis=1)  
#train['ANALWT_C']=np.log(train['ANALWT_C'])
y_pred = stack.fit_predict(train, target_train, test)        
y_pred[y_pred>.5]=1
y_pred[y_pred<=.5]=0
y_pred=y_pred.astype(int)
sample=pd.DataFrame()
sample['PERID']=id_test
sample['Criminal']=y_pred
sample.to_csv("D:/hackerearth/sample_submission_38.csv",index=False)
