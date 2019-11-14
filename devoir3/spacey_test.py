from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = English()

tokenizer = Tokenizer(nlp.vocab)

tokens = tokenizer('this ia , a sentence. Wher eyou are at?')
for token in tokens:
    print(token.norm_)
