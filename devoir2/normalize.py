from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

import sys

nlp = English()

tokenizer = Tokenizer(nlp.vocab)
for doc in tokenizer.pipe(sys.stdin):
    for token in doc:
        sys.stdout.write(token.text)
        sys.stdout.write('\n')
        sys.stdout.flush()
