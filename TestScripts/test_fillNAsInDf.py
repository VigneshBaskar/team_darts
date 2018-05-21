import pandas as pd
import os
from unittest import TestCase

from Scripts.FillNAsInDf import FillNAsInDf


class TestFillNAsInDf(TestCase):
    def test_fillnas(self):
        fill_nas_in_df_object = FillNAsInDf()
        df = pd.read_csv(os.path.join('Data', 'application_test.csv'))
        na_filled_df = fill_nas_in_df_object.fillnas(df)
        self.assertEqual(df.shape, na_filled_df.shape) and self.assertFalse(sum(df.isnull()))



