# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import pandas as pd
import numpy as np

# <codecell>

from Scripts.UnknownWordsProcessing import UnknownWordsProcessing 
from Scripts.VocabDict import VocabDict
from Scripts.MapWordToID import MapWordToID
from Scripts.Tokenizer import word_tokenizer
from Scripts.SentenceProcessing import SentenceProcessing

# <codecell>

from Scripts.Word2VecUtilities import Word2VecUtilities

# <codecell>

data_path = 'data'

# <codecell>

dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}

# <codecell>

train_csv = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype, low_memory=True)
train_csv = train_csv[pd.isnull(train_csv['project_essay_3'])]
train_csv['project_essay'] = train_csv['project_essay_1'] + train_csv['project_essay_2']
train_csv['project_essay'] = train_csv['project_essay'].str.lower()

y = train_csv['project_is_approved'].tolist()
X_text = train_csv['project_essay'].tolist()

# <codecell>

import pickle
data = {'X_text':X_text, 'y':y}
with open(os.path.join('Data','data.p'), 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# <codecell>

all_documents_tokenized_words = [list(set(word_tokenizer(text))) for text in X_text]
vocab_dict, rev_vocab_dict = VocabDict.create_vocab_dict(all_documents_tokenized_words, min_doc_count=1000)

# <codecell>

unknown_words_processing = UnknownWordsProcessing(vocab_dict.keys(), replace=False)
tokenized_documents = [word_tokenizer(text) for text in X_text]
unknown_words_removed_sentences = unknown_words_processing.remove_or_replace_unkown_word_from_sentences(tokenized_documents)
preprocessed_documents = SentenceProcessing().pad_truncate_sent(unknown_words_removed_sentences, chosen_sent_len = 300)

# <codecell>

w2v_model = Word2VecUtilities.create_word2vector_model(unknown_words_removed_sentences, wv_size=50)
embedding_matrix = Word2VecUtilities.create_embeddings_matrix(w2v_model, rev_vocab_dict)

# <codecell>

vocab_dict['my_dummy']=len(vocab_dict)
rev_vocab_dict[len(rev_vocab_dict)] = 'my_dummy'
embedding_matrix = np.vstack((embedding_matrix, np.zeros((1, embedding_matrix.shape[1]))))

# <codecell>

map_word_to_id = MapWordToID(vocab_dict)
id_lists = map_word_to_id.word_lists_to_id_lists(preprocessed_documents)
id_arrays = np.array(id_lists)

# <codecell>

def return_actual_text(x, rev_vocab_dict):
    actual_text = " ".join([rev_vocab_dict[word_id] for word_id in x])
    return actual_text

# <codecell>

w2v_model.wv.most_similar('grant')
