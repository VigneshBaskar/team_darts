# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

from sklearn.model_selection import train_test_split
import lightgbm as lgb

# <codecell>

application_train = pd.read_csv('Data/application_train.csv')

# <codecell>

pd.options.display.max_columns = len(application_train.columns)
pd.options.display.max_rows = len(application_train.columns)

# <codecell>

application_features = [column for column in application_train.columns.tolist() if column!='SK_ID_CURR' and column!='TARGET']

# <codecell>

categorical_features = []
continuous_features = []
for application_feature in application_features:
    if (application_train.dtypes[application_feature]=='object') or ( application_train.dtypes[application_feature]=='int64' and len(application_train[application_feature].unique())<20):
        categorical_features.append(application_feature)
    else:
        continuous_features.append(application_feature)

# <codecell>

# Replace Unknowns with 'unk' for categorical features and '0' for continuos features
application_train[categorical_features] = application_train[categorical_features].fillna(value='unk')
application_train[continuous_features] = application_train[continuous_features].fillna(value=0)

# <codecell>

for column in categorical_features:
    application_train[column] = application_train[column].astype('category')

# <codecell>

input_columns = application_train.columns
input_columns = input_columns[input_columns != 'TARGET']
target_column = 'TARGET'

X = application_train[input_columns]
y = application_train[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

lgb_train = lgb.Dataset(data=X_train, label=y_train)
lgb_eval = lgb.Dataset(data=X_test, label=y_test)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'num_iteration': 200,
        'verbose': 0
}

# train
gbm = lgb.train(params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)

# <codecell>

prob = 0.2

print(np.sum(gbm.predict(X_test)[np.asarray(y_test==1)]>prob)/np.sum(y_test==1))
print(np.sum(gbm.predict(X_test)[np.asarray(y_test==0)]<prob)/np.sum(y_test==0))
