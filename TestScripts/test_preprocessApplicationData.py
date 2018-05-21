import os
from unittest import TestCase

from Scripts.PreprocessApplicationData import PreprocessApplicationData


class TestPreprocessApplicationData(TestCase):
    def test_read_application_train_and_test_data(self):
        preprocess_application_data_object = PreprocessApplicationData()
        application_train, application_test = preprocess_application_data_object.read_application_train_and_test_data()
        self.assertLess(1000, len(application_train)) and self.assertLess(1000, len(application_test))

    def test_return_application_data(self):
        preprocess_application_data_object = PreprocessApplicationData()
        application_data = preprocess_application_data_object.return_application_data()
        self.assertLess(1000, len(application_data)) and self.assertEqual(123, len(application_data.columns))

