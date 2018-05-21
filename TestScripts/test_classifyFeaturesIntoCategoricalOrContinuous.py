import os
import pandas as pd
from unittest import TestCase

from Scripts.ClassifyFeaturesIntoCategoricalOrContinuous import ClassifyFeaturesIntoCategoricalOrContinuous



class TestClassifyFeaturesIntoCategoricalOrContinuous(TestCase):
    def test_classify(self):
        classify_object = ClassifyFeaturesIntoCategoricalOrContinuous()
        application_test = pd.read_csv(os.path.join('Data', 'application_test.csv'))
        classified_features_dict = classify_object.classify(application_test)
        categorical_features = classified_features_dict['categorical_features']
        continuous_features = classified_features_dict['continuous_features']
        self.assertTrue(True if 'FLAG_OWN_REALTY' in categorical_features else False) and  self.assertTrue(True if 'AMT_ANNUITY' in
                                                                                                                   continuous_features else False)







