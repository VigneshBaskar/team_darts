# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd

# <codecell>

application_train = pd.read_csv('Data/application_train.csv')

# <codecell>

pd.options.display.max_columns = len(application_train.columns)
pd.options.display.max_rows = len(application_train.columns)

# <codecell>

application_train.head()

# <codecell>

categorical_features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
                       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT',
                       'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                       'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
                       'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE']

# <codecell>

continuous_features = ['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE',
                      'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH','DAYS_EMPLOYED', 'DAYS_REGISTRATION','OWN_CAR_AGE',
                      'CNT_FAM_MEMBERS','HOUR_APPR_PROCESS_START','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3',
                      'APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG']

# <codecell>

for categorical_feature in categorical_features:
    print(categorical_feature, len(application_train[categorical_feature].unique()), application_train.dtypes[categorical_feature])

# <codecell>

for continuous_feature in continuous_features:
    print(continuous_feature, len(application_train[continuous_feature].unique()), application_train.dtypes[continuous_feature])
