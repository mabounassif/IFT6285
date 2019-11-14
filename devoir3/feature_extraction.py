import pandas as pd
import sys
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner'])

from sklearn.dummy import DummyClassifier 
from sklearn.metrics import classification_report 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC

import syllapy
import multiprocessing

NUM_OF_PROCESSES = multiprocessing.cpu_count()

df_train = pd.read_csv(sys.argv[1], names=['blog', 'class'])

def extract_features(data):
    def extract_from_blog(b):
        tokens = nlp(b)

        syl_1_count = 0
        syl_2_count = 0
        syl_3_count = 0
        syl_4_count = 0
        syl_5_count = 0
        syl_6_count = 0
        syl_7_count = 0
        syl_8_more_count = 0

        total_syl_count = 0
        total_char_count = 0
        total_complex_count = 0
        total_comma_count = 0
            
        complex_words_count = 0

        words_count = 0
        sentences_count = 0

        for s in tokens.sents:
            sentences_count += 1

            for w in tokens[s.start:s.end]:
                syl_count = syllapy.count(w.text)
                words_count += 1

                total_syl_count += syl_count
                total_char_count += len(w)

                if syl_count == 1:
                    syl_1_count += 1
                elif syl_count == 2:
                    syl_2_count += 1
                elif syl_count == 3:
                    syl_3_count += 1
                elif syl_count == 4:
                    syl_4_count += 1
                elif syl_count == 5:
                    syl_5_count += 1
                elif syl_count == 6:
                    syl_6_count += 1
                elif syl_count == 7:
                    syl_7_count += 1
                elif syl_count >= 8:
                    syl_8_more_count += 1

                if syl_count > 3 or len(w) > 13:
                    total_complex_count += 1

                if w.text == ',':
                    total_comma_count += 1

        return [
                total_char_count / words_count,
                words_count / sentences_count,
                total_complex_count / words_count,
                total_complex_count / sentences_count,
                total_syl_count / words_count,
                total_comma_count / sentences_count,
                syl_1_count / words_count,
                syl_2_count / words_count,
                syl_3_count / words_count,
                syl_4_count / words_count,
                syl_5_count / words_count,
                syl_6_count / words_count,
                syl_7_count / words_count,
                syl_8_more_count / words_count
                ]


    features = data.map(extract_from_blog)
    return features.tolist()

#s = "this is a sentence. this is a sentence, this."
#print(syllapy.count('this'))
#print(s)
#print(extract_features(pd.Series([s])))

pool = multiprocessing.Pool(NUM_OF_PROCESSES)
data_split = np.array_split(df_train['blog'], NUM_OF_PROCESSES)
data = np.concatenate(pool.map(extract_features, data_split))
pool.close()
pool.join()
print('Success')

np.savetxt("output-%s" % sys.argv[1], data, delimiter=',')
