



class ClassifyFeaturesIntoCategoricalOrContinuous(object):
    def classify(self, df):
        categorical_features = []
        continuous_features = []
        dataframe_columns = [column for column in df.columns.tolist() if column!='SK_ID_CURR' and column!='TARGET']
        for feature in dataframe_columns:
            if(df.dtypes[feature]=='object') or (df.dtypes[feature]=='int64' and len(df[feature].unique())<10):
                categorical_features.append(feature)
            else:
                continuous_features.append(feature)
        return {'categorical_features':categorical_features, 'continuous_features':continuous_features}
