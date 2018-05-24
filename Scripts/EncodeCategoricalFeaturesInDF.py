from sklearn.preprocessing import LabelEncoder

from Scripts.ClassifyFeaturesIntoCategoricalOrContinuous import ClassifyFeaturesIntoCategoricalOrContinuous
from Scripts.FillNAsInDf import FillNAsInDf


class EncodeCategoricalFeaturesInDF(object):
    def encode_features(self, df=None):
        le = LabelEncoder()
        classify_object = ClassifyFeaturesIntoCategoricalOrContinuous()
        classified_features_dict = classify_object.classify(df)
        categorical_features = classified_features_dict['categorical_features']
        fill_nas_in_df_object = FillNAsInDf()
        df = fill_nas_in_df_object.fillnas(df)
        for feature in categorical_features:
            df[feature] = le.fit_transform(df[feature])
        return df
