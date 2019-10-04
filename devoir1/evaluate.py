import kenlm
import operator
import sys
import argparse
import itertools
import multiprocessing

NUM_OF_PROCESSES = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description='Evaluate language models')
parser.add_argument('language', help='language to evaluate')

args = parser.parse_args()

if not args.language in ['en', 'fr', 'de']:
    sys.stdout.write('Error: language must be en, fr or de')
    sys.stdout.write("\n")
    sys.exit()

vocab = set(word for line in open('processed/train.' + args.language + '.vocab') for word in line.strip().split())
sys.stdout.write("loaded vocab with %s unique words" % (len(vocab)))
sys.stdout.write("\n")

model = kenlm.Model('models/train.' + args.language + '.binary')
sys.stdout.write("loaded %s-gram language model" % (model.order))
sys.stdout.write("\n")

total_word_count = 0.0
best_1_count = 0.0
best_3_count = 0.0
best_5_count = 0.0

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

def get_counts(line):
    sentence = ''
    state_in, state_out = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state_in)
    total_score = 0.0
    total_word_count = 0.0
    total_word_predicted = 0.0
    best_1_count = 0.0
    best_3_count = 0.0
    best_5_count = 0.0
                            
    line_a = line.split()

    for idx, word in enumerate(line_a):
        sentence += ' ' + word
        sentence = sentence.strip()
        #print('sentence: %s' % sentence)

        total_score += model.BaseScore(state_in, word, state_out)
        candidates = list((model.score(sentence + ' ' + next_word), next_word) for next_word in vocab)
        bad_words = sorted(candidates, key=operator.itemgetter(0), reverse=False)
        top_words = sorted(candidates, key=operator.itemgetter(0), reverse=True)

        worst_5 = bad_words[:5]
        #print('worst: %s' % (worst_5))

        best_5 = top_words[:5]
        best_3 = top_words[:3]
        best_1 = top_words[0]
        #print('best: %s' % (best_5))

        state_in, state_out = state_out, state_in

        if idx < len(line_a) - 1:
            if line_a[idx+1] in [best5[1] for best5 in best_5]:
                best_5_count += 1

            if line_a[idx+1] in [best3[1] for best3 in best_3]:
                best_3_count += 1

            if line_a[idx+1] ==  best_1[1]:
                best_1_count += 1

            total_word_predicted += 1

        total_word_count += 1


    return (total_score, total_word_count, best_1_count, best_3_count, best_5_count, total_word_predicted)


def worker(*lines):
    return [get_counts(line) for line in lines]

pool = multiprocessing.Pool(NUM_OF_PROCESSES)
results = []

for lines in chunk(sys.stdin, 500):
    result = pool.apply_async(worker, lines)
    results.append(result)

pool.close()
pool.join()
print('Success')

total_score = 0.0
total_word_count = 0.0
best_1_count = 0.0
best_3_count = 0.0
best_5_count = 0.0
total_word_predicted = 0.0

for r in results:
    for result in r.get():
        print(result)
        total_score += result[0]
        total_word_count += result[1]
        best_1_count += result[2]
        best_3_count += result[3]
        best_5_count += result[4]
        total_word_predicted += result[5]

sys.stdout.write("total_score: %s, total_word_count: %s, top_1: %s, top_3: %s, top_5: %s, total_words: %s" % (total_score, total_word_count, best_1_count, best_3_count, best_5_count, total_word_predicted))
sys.stdout.write("\n")
sys.stdout.flush()
            
