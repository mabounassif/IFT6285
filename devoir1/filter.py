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

def filter(word):
    return word if word in vocab else 'UNK'

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

def get_counts(line):
    line_a = line.split()

    sys.stdout.write(' '.join([filter(word) for idx, word in enumerate(line_a)]))
    sys.stdout.write('\n')



def worker(*lines):
    return [get_counts(line) for line in lines]

pool = multiprocessing.Pool(NUM_OF_PROCESSES)
results = []

for lines in chunk(sys.stdin, 500):
    result = pool.apply_async(worker, lines)
    results.append(result)

pool.close()
pool.join()
