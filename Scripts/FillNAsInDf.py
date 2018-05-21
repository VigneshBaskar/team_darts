from Scripts.ClassifyFeaturesIntoCategoricalOrContinuous import ClassifyFeaturesIntoCategoricalOrContinuous


class FillNAsInDf(object):
    def fillnas(self, df):
        classify_object = ClassifyFeaturesIntoCategoricalOrContinuous()
        classified_features_dict = classify_object.classify(df)
        categorical_features = classified_features_dict['categorical_features']
        continuous_features = classified_features_dict['continuous_features']
        df[categorical_features] = df[categorical_features].fillna(value='unk')
        df[continuous_features] = df[continuous_features].fillna(value=0)
        for column in categorical_features:
            df[column] = df[column].astype('category')
        return df






