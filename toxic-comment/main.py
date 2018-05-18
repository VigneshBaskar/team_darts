# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import pandas as pd
import numpy as np

# <codecell>

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
subm = pd.read_csv('Data/sample_submission.csv')

# <codecell>

# Adding a label clean
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['clean'] = train[label_cols].max(axis=1)

# Replacing empty cells with text unknown
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)
