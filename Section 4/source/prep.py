"""
Data available from http://www.manythings.org/anki/
under eng-spa.zip link.

Inspired by:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Before you can use this script you need to install a unidecode package with:
pip3 install unidecode
"""

import random
import unidecode

SOS_token = 0
EOS_token = 1
UW_token = 2

MAX_SENTENCE_LENGTH = 10

class Vocab:
    """
    Keeping track of language vocabulary.
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UW"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        """
        Add each word from sentence.
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Add a new word to a vocabulary,
        update all the counters and indexes.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def qscleaner(w):
    """
    Just remove ? character from a word.
    """
    w=w.replace('?','')
    return w

def isquestion(s, max_length=MAX_SENTENCE_LENGTH):
    """
    Return True if sentence is valid according
    to our criteria.

    Here we're interested in questions that are
    no longer than mex_legth.
    """
    return len(s.split(' ')) < max_length and len(s.split(' ')) < max_length and s.find('?') != -1

def clean(s, extra_cleaner=qscleaner):
    """
    Clean up the whole sentence:
    Include only words, make
    them lower case and
    remove any non-english characters.
    """
    include_words=[]
    for word in s.split():
        word=word.strip().lower()
        word=unidecode.unidecode(word)
        word=extra_cleaner(word)
        if word.isdigit():
            continue
        include_words.append(word)
    return ' '.join(include_words)

def process_file(ilang, olang, limit, sfilter=isquestion):
    """
    Read a language file, clean up sentences
    and based on them create a Vocab object for
    each language, return only limit sentences.
    """
    print("Reading sentences...")
    sentences = open('data/%s.txt' % olang, encoding='utf-8').read().splitlines()
    pairs = [[clean(w) for w in s.split('\t')] for s in sentences if sfilter(s)]
    pairs = [list(p) for p in pairs]
    input_lang = Vocab(ilang)
    output_lang = Vocab(olang)
    return pairs, input_lang, output_lang

def get_data(ilang, olang, limit=100, log=print):
    """
    Return a limit number of sentences of both ilang and olang.
    Sentences has to match criteria defined by sfilter and
    are processed by wclean.

    ilang - input language that we want to translate from

    olang - output language that we want to tranlate to

    limit - a number of sentences to process for each language
            choose small number if you don't have GPU processing power
    """
    pairs, input_lang, output_lang = process_file(ilang, olang, limit)
    log("Got %d sentences in both langs" % len(pairs))
    pairs = [ pair for pair in pairs if pair][:limit]
    log("Reduced to %d sentences" % len(pairs))
    log("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    log("Counted words:")
    log(input_lang.name, input_lang.n_words)
    log(output_lang.name, output_lang.n_words)
    log('Random data sample:')
    log(random.choice(pairs))
    return pairs,input_lang, output_lang

if __name__ == '__main__':
    p, il, ol=get_data('en','spa')
    from pprint import pprint
    pprint(('word2index', ol.word2index))
    pprint(('word2count', ol.word2count))
    pprint(('index2word', ol.index2word))
