import os
import pandas as pd


class PreprocessApplicationData(object):
    def __init__(self):
        application_train, application_test = self.read_application_train_and_test_data()
        self.application_data = pd.concat([application_train, application_test])

    def read_application_train_and_test_data(self):
        print(os.getcwd())
        application_train = pd.read_csv(os.path.join('Data','application_train.csv'))
        application_test = pd.read_csv(os.path.join('Data','application_test.csv'))
        # Add an additional column to split in the future
        application_train['data_type'] = 'train'
        application_test['data_type'] = 'test'
        return application_train, application_test


