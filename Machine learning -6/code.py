# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 00:44:49 2018

@author: Karra's
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 23:35:20 2018

@author: Karra's
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

from sklearn.model_selection import KFold
train=pd.read_csv("D:/hackerearth/ml_6new/Dataset/train.csv")
test=pd.read_csv("D:/hackerearth/ml_6new/Dataset/test.csv")
submission=pd.read_csv("D:/hackerearth/ml_6new/Dataset/sample_submission.csv")
structure=pd.read_csv("D:/hackerearth/ml_6new/Dataset/Building_Structure.csv")
ownership=pd.read_csv("D:/hackerearth/ml_6new/Dataset/Building_Ownership_Use.csv")

h=test[test['area_assesed']=='Building removed'].index.tolist()

train["damage_grade"]=train["damage_grade"].replace({"Grade 1":1,"Grade 2":2,"Grade 3":3,"Grade 4":4,"Grade 5":5})

y=train["damage_grade"]


#train.drop(['damage_grade'],axis=1,inplace=True)
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
      # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)              
 # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist               

train, NAlist_train = reduce_mem_usage(train)
ownership, NAlist_ownership = reduce_mem_usage(ownership)
structure, NAlist_structure = reduce_mem_usage(structure)
test, NAlist_test = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist_train)
print(NAlist_structure)
print(NAlist_ownership)
print(NAlist_test)

#train=train.replace(255,-1)
#test=test.replace(255,-1)

def getCountVar(compute_df, count_df, var_name):
        grouped_df = count_df.groupby(var_name)
        count_dict = {}
        for name, group in grouped_df:
                count_dict[name] = group.shape[0]

        count_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                count_list.append(count_dict.get(name, 0))
        return count_list

def getPurchaseVar(compute_df, purchase_df, var_name):
        grouped_df = purchase_df.groupby(var_name)
        min_dict = {}
        max_dict = {}
        mean_dict = {}
       # twentyfive_dict = {}
       # seventyfive_dict = {}
        for name, group in grouped_df:
                min_dict[name] = min(np.array(group["damage_grade"]))
                max_dict[name] = max(np.array(group["damage_grade"]))
                mean_dict[name] = np.mean(np.array(group["damage_grade"]))
           #     twentyfive_dict[name] = np.percentile(np.array(group["damage_grade"]),25)
            #    seventyfive_dict[name] = np.percentile(np.array(group["damage_grade"]),75)

        min_list = []
        max_list = []
        mean_list = []
       # twentyfive_list = []
       # seventyfive_list = []
        for index, row in compute_df.iterrows():
                name = row[var_name]
                min_list.append(min_dict.get(name,0))
                max_list.append(max_dict.get(name,0))
                mean_list.append(mean_dict.get(name,0))
             #   twentyfive_list.append( twentyfive_dict.get(name,0))
              #  seventyfive_list.append( seventyfive_dict.get(name,0))

        return min_list, max_list, mean_list
    
train = pd.merge(train ,ownership, on='building_id', how='left', suffixes=('_','')).fillna(-1)
train.drop(['vdcmun_id_','district_id_'], axis=1, inplace=True)
train = pd.merge(train ,structure, on='building_id', how='left', suffixes=('_','')).fillna(-1)
train.drop(['vdcmun_id_','district_id_','ward_id_'], axis=1, inplace=True)


test = pd.merge(test ,ownership, on='building_id', how='left', suffixes=('_','')).fillna(-1)
test.drop(['vdcmun_id_','district_id_'], axis=1, inplace=True)
test = pd.merge(test ,structure, on='building_id', how='left', suffixes=('_','')).fillna(-1)
test.drop(['vdcmun_id_','district_id_','ward_id_'], axis=1, inplace=True)

cols = ['count_floors_pre_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq',
       'land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration','legal_ownership_status', 'count_families','damage_grade']
import matplotlib.pyplot as plt
for col in cols:
    print("==========================================================")
    #print(f"\t \t{col}")
    train[col].hist()
    plt.show()
    
train_1=train[train['damage_grade']==1]
train_5=train[train['damage_grade']==5]
train_1_drop=train[train['damage_grade']!=1]
train_1_5_drop=train_1_drop[train_1_drop['damage_grade']!=5]
p=train.columns.difference(['district_id','ward_id','vdcmun_id','building_id','damage_grade'])
train_1_5_drop.drop_duplicates(subset=p,keep=False,inplace=True)
train_=pd.concat([train_1_5_drop,train_1])
train=pd.concat([train_,train_5])
y=train["damage_grade"]
    
train["District_ID_Count"]=getCountVar(train, train, "district_id")
test["District_ID_Count"]=getCountVar(test, train, "district_id")
min_price_list, max_price_list, mean_price_list = getPurchaseVar(train, train, "district_id")
train["District_ID_MinGrade"] = min_price_list
train["District_ID_MaxGrade"] = max_price_list
#train["District_ID_MeanGrade"] = mean_price_list
min_price_list, max_price_list, mean_price_list = getPurchaseVar(test, train, "district_id")
test["District_ID_MinGrade"] = min_price_list
test["District_ID_MaxGrade"] = max_price_list
#test["District_ID_MeanGrade"] = mean_price_list

train["Ward_ID_Count"]=getCountVar(train, train, "ward_id")
test["Ward_ID_Count"]=getCountVar(test, train, "ward_id")
min_price_list, max_price_list, mean_price_list = getPurchaseVar(train, train, "ward_id")
train["Ward_ID_MinGrade"] = min_price_list
train["Ward_ID_MaxGrade"] = max_price_list
#train["Ward_ID_MeanGrade"] = mean_price_list
min_price_list, max_price_list, mean_price_list = getPurchaseVar(test, train, "ward_id")
test["Ward_ID_MinGrade"] = min_price_list
test["Ward_ID_MaxGrade"] = max_price_list
#test["Ward_ID_MeanGrade"] = mean_price_list

train["vdcmun_ID_Count"]=getCountVar(train, train, "vdcmun_id")
test["vdcmun_ID_Count"]=getCountVar(test, train, "vdcmun_id")
min_price_list, max_price_list, mean_price_list = getPurchaseVar(train, train, "vdcmun_id")
train["vdcmun_ID_MinGrade"] = min_price_list
train["vdcmun_ID_MaxGrade"] = max_price_list
#train["vdcmun_ID_MeanGrade"] = mean_price_list
min_price_list, max_price_list, mean_price_list = getPurchaseVar(test, train, "vdcmun_id")
test["vdcmun_ID_MinGrade"] = min_price_list
test["vdcmun_ID_MaxGrade"] = max_price_list
#test["vdcmun_ID_MeanGrade"] = mean_price_list


#train['height_increased']=train['height_pre_eq']<train['height_post_eq']
np.random.seed(13)

def impact_coding(data, feature, target='damage_grade'):
    '''
    In this implementation we get the values and the dictionary as two different steps.
    This is just because initially we were ignoring the dictionary as a result variable.
    
    In this implementation the KFolds use shuffling. If you want reproducibility the cv 
    could be moved to a parameter.
    '''
    n_folds = 10
    n_inner_folds = 5
    impact_coded = pd.Series()
    
    oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
    kf = KFold(n_splits=n_folds, shuffle=True)
    oof_mean_cv = pd.DataFrame()
    split = 0
    for infold, oof in kf.split(data[feature]):
            impact_coded_cv = pd.Series()
            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            inner_oof_mean_cv = pd.DataFrame()
            oof_default_inner_mean = data.iloc[infold][target].mean()
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
                oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
                impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                            lambda x: oof_mean[x[feature]]
                                      if x[feature] in oof_mean.index
                                      else oof_default_inner_mean
                            , axis=1))

                # Also populate mapping (this has all group -> mean for all inner CV folds)
                inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
                inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
                inner_split += 1

            # Also populate mapping
            oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
            oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
            split += 1
            
            impact_coded = impact_coded.append(data.iloc[oof].apply(
                            lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
                                      if x[feature] in inner_oof_mean_cv.index
                                      else oof_default_mean
                            , axis=1))

    return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean



impact_coding_map = {}
categorical_features=['district_id','ward_id','vdcmun_id','plan_configuration','condition_post_eq','area_assesed','ground_floor_type','other_floor_type','foundation_type','legal_ownership_status','land_surface_condition','roof_type']
for f in categorical_features:
    print("Impact coding for {}".format(f))
    train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f)
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1)

#import matplotlib.pyplot as plt    
#corrmat = train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True)                    

def getDVEncodeVar(compute_df, target_df, var_name, target_var="damage_grade", min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["mean"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])


'''
train['district_vdcmun_mean']=getDVEncodeVar(train, train, ['district_id','vdcmun_id'], 'damage_grade')
test['district_vdcmun_mean']=getDVEncodeVar(test, train, ['district_id','vdcmun_id'], 'damage_grade')
'''


def getDVEncodeVar_2(compute_df, target_df, var_name, target_var, min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["count"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])
'''
train['buildings_in_district']=getDVEncodeVar_2(train, train, ['district_id'], 'building_id')
test['buildings_in_district']=getDVEncodeVar_2(test, train, ['district_id'], 'building_id')

train['buildings_in_ward']=getDVEncodeVar_2(train, train, ['ward_id'], 'building_id')
test['buildings_in_ward']=getDVEncodeVar_2(test, train, ['ward_id'], 'building_id')

train['buildings_in_vdcmun']=getDVEncodeVar_2(train, train, ['vdcmun_id'], 'building_id')
test['buildings_in_vdcmun']=getDVEncodeVar_2(test, train, ['vdcmun_id'], 'building_id')

train['height_by_area_post']=train['height_ft_post_eq']/(train['plinth_area_sq_ft']+.001)
test['height_by_area_post']=test['height_ft_post_eq']/(test['plinth_area_sq_ft']+.001)


train['height_by_area_pre']=train['height_ft_pre_eq']/(train['plinth_area_sq_ft']+.001)
test['height_by_area_pre']=test['height_ft_pre_eq']/(test['plinth_area_sq_ft']+.001)
'''
#df = pd.concat([train ,pd.get_dummies(train['damage_grade'],prefix='grade')], axis=1)


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
#train['age_building']=pd.qcut(train['age_building'],4)
#train['age_building']=label.fit_transform(train['age_building'])
#test['age_building']=label.fit_transform(test['age_building'])


train['height_diff']=train['height_ft_pre_eq']-train['height_ft_post_eq']
test['height_diff']=test['height_ft_pre_eq']-test['height_ft_post_eq']

train['floor_diff']=train['count_floors_pre_eq']-(train['count_floors_post_eq'])
test['floor_diff']=test['count_floors_pre_eq']-(test['count_floors_post_eq'])


train['height_ratio']=train['height_ft_post_eq']/(train['height_ft_pre_eq']+.001)
test['height_ratio']=test['height_ft_post_eq']/(test['height_ft_pre_eq']+.001)

train['floor_ratio']=train['count_floors_post_eq']/(train['count_floors_pre_eq']+.001)
test['floor_ratio']=test['count_floors_post_eq']/(test['count_floors_pre_eq']+.001)


#ulimit = np.percentile(train.height_diff.values, 99)
#train['height_diff'].ix[train['height_diff']>ulimit] = ulimit

#ulimit = np.percentile(train.floor_diff.values, 99)
#train['floor_diff'].ix[train['floor_diff']>ulimit] = ulimit

def encode_count(df,column_name):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

for col in train.columns.difference(['building_id']):
        if train[col].dtype == object:
            train=encode_count(train,col)

for col in test.columns.difference(['building_id']):
        if test[col].dtype == object:
            test=encode_count(test,col)
  
full=pd.concat([train['building_id'],test['building_id']]) 
full=pd.DataFrame(full)         
full=encode_count(full,'building_id')           
train['building_id']=full[:train.shape[0]]
test['building_id']=full[train.shape[0]:]

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#df = pd.concat([train ,pd.get_dummies(train['damage_grade'],prefix='grade')], axis=1)

'''
categorical_features=['district_id','ward_id','vdcmun_id']
grades=['grade_1','grade_2','grade_3','grade_4','grade_5']
for f in categorical_features:
    for n in grades:
        print("Impact coding for {}".format(f))
        train["impact_encoded_{}_{}".format(f,n)], impact_coding_mapping, default_coding = impact_coding(train, f,n)
        impact_coding_map[f] = (impact_coding_mapping, default_coding)
        mapping, default_mean = impact_coding_map[f]
        test["impact_encoded_{}_{}".format(f,n)] = test.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1)  
     
    
'''
train['risk_severity']=train['has_geotechnical_risk_fault_crack']+train['has_geotechnical_risk_flood']+train['has_geotechnical_risk_land_settlement']+train['has_geotechnical_risk_landslide']+train['has_geotechnical_risk_liquefaction']+train['has_geotechnical_risk_other']+train['has_geotechnical_risk_rock_fall']
test['risk_severity']=test['has_geotechnical_risk_fault_crack']+test['has_geotechnical_risk_flood']+test['has_geotechnical_risk_land_settlement']+test['has_geotechnical_risk_landslide']+test['has_geotechnical_risk_liquefaction']+test['has_geotechnical_risk_other']+test['has_geotechnical_risk_rock_fall']


train['secondary_use_severity']=train['has_secondary_use_agriculture']+train['has_secondary_use_hotel']+train['has_secondary_use_rental']+train['has_secondary_use_institution']+train['has_secondary_use_rental']+train['has_secondary_use_industry']+train['has_secondary_use_health_post']+train['has_secondary_use_gov_office']+train['has_secondary_use_use_police']
test['secondary_use_severity']=test['has_secondary_use_agriculture']+test['has_secondary_use_hotel']+test['has_secondary_use_rental']+test['has_secondary_use_institution']+test['has_secondary_use_rental']+test['has_secondary_use_industry']+test['has_secondary_use_health_post']+test['has_secondary_use_gov_office']+test['has_secondary_use_use_police']


train['super_structure_rate']=train['has_superstructure_adobe_mud']+train['has_superstructure_mud_mortar_stone']+train['has_superstructure_stone_flag']+train['has_superstructure_adobe_mud']+train['has_superstructure_cement_mortar_stone']+train['has_superstructure_cement_mortar_brick']+train['has_superstructure_timber']+train['has_superstructure_bamboo']+train['has_superstructure_rc_non_engineered']+train['has_superstructure_rc_engineered']+train['has_superstructure_other']
test['super_structure_rate']=test['has_superstructure_adobe_mud']+test['has_superstructure_mud_mortar_stone']+test['has_superstructure_stone_flag']+test['has_superstructure_adobe_mud']+test['has_superstructure_cement_mortar_stone']+test['has_superstructure_cement_mortar_brick']+test['has_superstructure_timber']+test['has_superstructure_bamboo']+test['has_superstructure_rc_non_engineered']+test['has_superstructure_rc_engineered']+test['has_superstructure_other']

#train['height_inc'] = train.apply(lambda x : 1 if x['height_ft_post_eq'] >= x['height_ft_pre_eq']  2 else if x['height_ft_post_eq'] < x['height_ft_pre_eq']else  0, axis=1)


train['geotechnical_risk']=train['has_geotechnical_risk_fault_crack'].astype(str)+'_'+train['has_geotechnical_risk_flood'].astype(str)+'_'+train['has_geotechnical_risk_land_settlement'].astype(str)+'_'+train['has_geotechnical_risk_landslide'].astype(str)+'_'+train['has_geotechnical_risk_liquefaction'].astype(str)+'_'+train['has_geotechnical_risk_other'].astype(str)+'_'+train['has_geotechnical_risk_rock_fall'].astype(str)
test['geotechnical_risk']=test['has_geotechnical_risk_fault_crack'].astype(str)+'_'+test['has_geotechnical_risk_flood'].astype(str)+'_'+test['has_geotechnical_risk_land_settlement'].astype(str)+'_'+test['has_geotechnical_risk_landslide'].astype(str)+'_'+test['has_geotechnical_risk_liquefaction'].astype(str)+'_'+test['has_geotechnical_risk_other'].astype(str)+'_'+test['has_geotechnical_risk_rock_fall'].astype(str)

train['secondary_use']=train['has_secondary_use_agriculture'].astype(str)+'_'+train['has_secondary_use_hotel'].astype(str)+'_'+train['has_secondary_use_rental'].astype(str)+'_'+train['has_secondary_use_institution'].astype(str)+'_'+train['has_secondary_use_rental'].astype(str)+'_'+train['has_secondary_use_industry'].astype(str)+'_'+train['has_secondary_use_health_post'].astype(str)+'_'+train['has_secondary_use_gov_office'].astype(str)+'_'+train['has_secondary_use_use_police'].astype(str)
test['secondary_use']=test['has_secondary_use_agriculture'].astype(str)+'_'+test['has_secondary_use_hotel'].astype(str)+'_'+test['has_secondary_use_rental'].astype(str)+'_'+test['has_secondary_use_institution'].astype(str)+'_'+test['has_secondary_use_rental'].astype(str)+'_'+test['has_secondary_use_industry'].astype(str)+'_'+test['has_secondary_use_health_post'].astype(str)+'_'+test['has_secondary_use_gov_office'].astype(str)+'_'+test['has_secondary_use_use_police'].astype(str)


train['super_structure']=train['has_superstructure_adobe_mud'].astype(str)+'_'+train['has_superstructure_mud_mortar_stone'].astype(str)+'_'+train['has_superstructure_stone_flag'].astype(str)+'_'+train['has_superstructure_adobe_mud'].astype(str)+'_'+train['has_superstructure_cement_mortar_stone'].astype(str)+'_'+train['has_superstructure_cement_mortar_brick'].astype(str)+'_'+train['has_superstructure_timber'].astype(str)+'_'+train['has_superstructure_bamboo'].astype(str)+'_'+train['has_superstructure_rc_non_engineered'].astype(str)+'_'+train['has_superstructure_rc_engineered'].astype(str)+'_'+train['has_superstructure_other'].astype(str)
test['super_structure']=test['has_superstructure_adobe_mud'].astype(str)+'_'+test['has_superstructure_mud_mortar_stone'].astype(str)+'_'+test['has_superstructure_stone_flag'].astype(str)+'_'+test['has_superstructure_adobe_mud'].astype(str)+'_'+test['has_superstructure_cement_mortar_stone'].astype(str)+'_'+test['has_superstructure_cement_mortar_brick'].astype(str)+'_'+test['has_superstructure_timber'].astype(str)+'_'+test['has_superstructure_bamboo'].astype(str)+'_'+test['has_superstructure_rc_non_engineered'].astype(str)+'_'+test['has_superstructure_rc_engineered'].astype(str)+'_'+test['has_superstructure_other'].astype(str)

new_col=['geotechnical_risk','secondary_use','super_structure']
for i in new_col:
    train=encode_count(train,i);
    test=encode_count(test,i)
train["geotechnical_risk_count"]=getCountVar(train, train, "geotechnical_risk")
test["geotechnical_risk_count"]=getCountVar(test, train, "geotechnical_risk")


train["secondary_use_count"]=getCountVar(train, train, "secondary_use")
test["secondary_use_count"]=getCountVar(test, train, "secondary_use")

train["super_structure_count"]=getCountVar(train, train, "super_structure")
test["super_structure_count"]=getCountVar(test, train, "super_structure")



impact_coding_map = {}
new_categorical_features=['geotechnical_risk','secondary_use','super_structure']
for f in new_categorical_features:
    print("Impact coding for {}".format(f))
    train["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train, f)
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test["impact_encoded_{}".format(f)] = test.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1) 


train['height_diff']=train['height_ft_pre_eq']-train['height_ft_post_eq']
test['height_diff']=test['height_ft_pre_eq']-test['height_ft_post_eq']

train['floor_diff']=train['count_floors_pre_eq']-(train['count_floors_post_eq'])
test['floor_diff']=test['count_floors_pre_eq']-(test['count_floors_post_eq'])
'''
plt.rcParams.update({'font.size': 14})
sns.factorplot(x="damage_grade", y="has_geotechnical_risk_rock_fall", data=train_1_5_drop, kind="violin", size=15, aspect=.7);
'''
def que(x):
    if x['height_ft_post_eq'] > x['height_ft_pre_eq'] :
        return 1
    if x['height_ft_post_eq'] < x['height_ft_pre_eq'] :
        return 2
    else:
        return 3
        
train['height_inc'] = train.apply(que, axis=1)
test['height_inc'] = test.apply(que, axis=1)

def que(x):
    if x['count_floors_post_eq'] > x['count_floors_pre_eq'] :
        return 1
    if x['count_floors_post_eq'] < x['count_floors_pre_eq'] :
        return 2
    else:
        return 3
        
train['count_floors_inc'] = train.apply(que, axis=1)
test['count_floors_inc'] = test.apply(que, axis=1)

##########################################################
train=train[train['count_families']!=255]
train=train[train['count_families']!=11]
gmap={255:-1,1:1,0:0}
#train['has_repair_started']=train['has_repair_started'].map(gmap)
#test['has_repair_started']=test['has_repair_started'].map(gmap)
y=train['damage_grade']
train.loc[((train['height_ft_post_eq']-train['height_ft_pre_eq'])>50)&(train['height_ft_post_eq']!=0),'height_ft_post_eq']=train['height_ft_pre_eq']
test.loc[((test['height_ft_post_eq']-test['height_ft_pre_eq'])>50)&(test['height_ft_post_eq']!=0),'height_ft_post_eq']=test['height_ft_pre_eq']

'''
train['height_diff']=((train['height_ft_pre_eq'].astype(int))-(train['height_ft_post_eq'].astype(int))).astype(int)

test['height_diff']=((test['height_ft_pre_eq'].astype(int))-(test['height_ft_post_eq'].astype(int))).astype(int)

train['floor_diff']=((train['count_floors_pre_eq'].astype(int))-(train['count_floors_post_eq'].astype(int))).astype(int)

test['floor_diff']=((test['count_floors_pre_eq'].astype(int))-(test['count_floors_post_eq'].astype(int))).astype(int)

'''
#s=train.apply(lambda x:(x.max()))

###############################################################




#s=train.apply(lambda x:(x.max()))







#features_to_use.extend(['g5','g4','g3','g2','g1','w5','w4', 'w3','w2','w1','v5','v4', 'v3','v2','v1','a5','a4', 'a3','a2','a1'])
#test_x=test.drop(['building_id','district_id','ward_id','vdcmun_id'],axis=1)     


#these_features = [f for f in features_to_use if f not in ['damage_grade','grade_1','grade_2','grade_3','grade_4','grade_5']]

#train.to_pickle('D:/hackerearth/ml_6new/train.pkl')
#train=pd.read_pickle('D:/hackerearth/ml_6new/train.pkl')
#test.to_pickle('D:/hackerearth/ml_6new/test.pkl')
#these_features = [f for f in features_to_use if f not in ['building_id','damage_grade','District_ID_MaxGrade','District_ID_MinGrade','district_vdcmun_mean']]

'''
import pandas as pd
from collections import Counter

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}
'''
#class_weights = get_class_weights(y)
#print(class_weights)
#{1: 3.44, 2: 2.48, 3: 1.72, 4: 1.38, 5: 1.0}
from sklearn.ensemble import  BaggingClassifier

#X_train, X_val, y_train, y_val = train_test_split(train, y, test_size = 0.33,random_state=99)

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size = 0,random_state=99)

temp = pd.concat([X_train.district_id,pd.get_dummies(y_train)], axis = 1).groupby('district_id').mean()
temp.columns = ['g5','g4', 'g3','g2','g1']
temp['count'] = X_train.groupby('district_id').count().iloc[:,1]
 
print(temp.tail(10))

unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['g5','g4', 'g3','g2','g1']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['g5','g4', 'g3','g2','g1']] = mean_values.values
print(temp.tail(10))


X_train = X_train.merge(temp.reset_index(),how='left', left_on='district_id', right_on='district_id')


X_test=test

X_test = X_test.merge(temp.reset_index(),how='left', left_on='district_id', right_on='district_id')
new_manager_ixes = X_test['g5'].isnull()
X_test.loc[new_manager_ixes,['g5','g4', 'g3','g2','g1']] = mean_values.values
X_test.head()

temp = pd.concat([X_train.ward_id,pd.get_dummies(y_train)], axis = 1).groupby('ward_id').mean()
temp.columns = ['w5','w4', 'w3','w2','w1']
temp['count'] = X_train.groupby('ward_id').count().iloc[:,1]
 
print(temp.tail(10))

unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['w5','w4', 'w3','w2','w1']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['w5','w4', 'w3','w2','w1']] = mean_values.values
print(temp.tail(10))
X_train = X_train.merge(temp.reset_index(),how='left', left_on='ward_id', right_on='ward_id')

X_test = X_test.merge(temp.reset_index(),how='left', left_on='ward_id', right_on='ward_id')
new_manager_ixes = X_test['w5'].isnull()
X_test.loc[new_manager_ixes,['w5','w4', 'w3','w2','w1']] = mean_values.values
X_test.head()


temp = pd.concat([X_train.vdcmun_id,pd.get_dummies(y_train)], axis = 1).groupby('vdcmun_id').mean()
temp.columns = ['v5','v4', 'v3','v2','v1']
temp['count'] = X_train.groupby('vdcmun_id').count().iloc[:,1]
 
print(temp.tail(10))

unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['v5','v4', 'v3','v2','v1']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['v5','v4', 'v3','v2','v1']] = mean_values.values
print(temp.tail(10))


X_train = X_train.merge(temp.reset_index(),how='left', left_on='vdcmun_id', right_on='vdcmun_id')

X_test = X_test.merge(temp.reset_index(),how='left', left_on='vdcmun_id', right_on='vdcmun_id')
new_manager_ixes = X_test['v5'].isnull()
X_test.loc[new_manager_ixes,['v5','v4', 'v3','v2','v1']] = mean_values.values
X_test.head()

temp = pd.concat([X_train.has_repair_started,pd.get_dummies(y_train)], axis = 1).groupby('has_repair_started').mean()
temp.columns = ['r5','r4', 'r3','r2','r1']
temp['count'] = X_train.groupby('has_repair_started').count().iloc[:,1]
 
print(temp.tail(10))

unranked_managers_ixes = temp['count']<20
# ... and ranked ones
ranked_managers_ixes = ~unranked_managers_ixes

# compute mean values from ranked managers and assign them to unranked ones
mean_values = temp.loc[ranked_managers_ixes, ['r5','r4', 'r3','r2','r1']].mean()
print(mean_values)
temp.loc[unranked_managers_ixes,['r5','r4', 'r3','r2','r1']] = mean_values.values
print(temp.tail(10))
X_train = X_train.merge(temp.reset_index(),how='left', left_on='has_repair_started', right_on='has_repair_started')

X_test = X_test.merge(temp.reset_index(),how='left', left_on='has_repair_started', right_on='has_repair_started')
new_manager_ixes = X_test['r5'].isnull()
X_test.loc[new_manager_ixes,['r5','r4', 'r3','r2','r1']] = mean_values.values
X_test.head()

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
features_to_use=[]          
features_to_use.extend(X_train.columns)

these_features = [f for f in features_to_use if f not in ['damage_grade','District_ID_MaxGrade','District_ID_MinGrade','district_vdcmun_mean','a1','a2','a3','a4','a5','count_x','count_y','rf5','rf4', 'rf3','rf2','rf1','c5', 'c4', 'c3', 'c2', 'c1','ag5','ag4', 'ag3','ag2','ag1','l1','l2','l3','l4','l5','count','impact_encoded_position']]

clf = RandomForestClassifier(n_estimators=200,min_samples_leaf=5,random_state=99)
clfbag = BaggingClassifier(clf,n_estimators=5,random_state=99)
clfbag.fit(X_train[these_features], y_train)
#y_val_pred = clf.predict_proba(X_val[these_features])


#log_loss(y_val, y_val_pred)


y_pred = clfbag.predict(X_test[these_features])

submission['damage_grade']=y_pred
submission['damage_grade']=submission['damage_grade'].replace({1:"Grade 1",2:"Grade 2",3:"Grade 3",4:"Grade 4",5:"Grade 5"})

submission.to_csv("D:/hackerearth/ml_6new/Dataset/new_sol.csv",index=False)