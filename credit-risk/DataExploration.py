# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightclf as lgb
from sklearn.metrics import roc_auc_score

# <codecell>

application_train = pd.read_csv('Data/application_train.csv')
application_test = pd.read_csv('Data/application_test.csv')

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

application_test[categorical_features] = application_test[categorical_features].fillna(value='unk')
application_test[continuous_features] = application_test[continuous_features].fillna(value=0)

# <codecell>

for column in categorical_features:
    application_train[column] = application_train[column].astype('category')
    application_test[column] = application_test[column].astype('category')

# <codecell>

input_columns = categorical_features + continuous_features
target_column = 'TARGET'

X = application_train[input_columns]
y = application_train[target_column]
X_test = application_test[input_columns]

# <codecell>

merged_X = pd.concat([X, X_test])

# <codecell>

le = LabelEncoder()
for feature in categorical_features:
    merged_X[feature] = le.fit_transform(merged_X[feature])

# <codecell>

X = merged_X[:len(X)]
X_test = merged_X[len(X):]

# <codecell>

seed = 1234
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=seed)

# <codecell>

scale_pos_weight = sum(application_train.TARGET==1)/sum(application_train.TARGET==0)
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, scale_pos_weight=scale_pos_weight).fit(X_train.as_matrix(), y_train.as_matrix())
predictions = clf.predict_proba(X_valid.as_matrix())[:,1]

# <codecell>

print('Area Under Curve:',roc_auc_score(y_valid.as_matrix(), predictions))

# <codecell>

test_predictions = clf.predict_proba(X_test.as_matrix())[:,1]
xgb_test_predictions = pd.DataFrame({'SK_ID_CURR':application_test['SK_ID_CURR'], 'TARGET':test_predictions})
xgb_test_predictions.to_csv('Data/xgb_submission.csv', index=False, float_format='%.8f')
