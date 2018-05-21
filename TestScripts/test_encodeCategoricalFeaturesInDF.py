import os
import pandas as pd
from unittest import TestCase

from Scripts.EncodeCategoricalFeaturesInDF import EncodeCategoricalFeaturesInDF


class TestEncodeCategoricalFeaturesInDF(TestCase):
    def test_encode_features(self):
        encoding_object = EncodeCategoricalFeaturesInDF()
        application_test = pd.read_csv(os.path.join('Data', 'application_test.csv'))
        application_test_encoded_df = encoding_object.encode_features(application_test)
        self.assertEqual(application_test.shape, application_test_encoded_df.shape)
