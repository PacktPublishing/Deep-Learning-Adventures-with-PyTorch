"""
Prepare language data for training language detector.

We're using a language wikipedia data set downloaded from:
https://zenodo.org/record/841984#.W9HFHS97FQI

To convert language data to ascii we use unidecode package
that you can install using pip:
pip3 install unidecode

Before using scripts from this section make sure to
set the correct output encoding in your terminal
(this is needed since we're working with a variety of
 languages and text encodings):
export PYTHONIOENCODING="UTF-8"
"""
import unidecode

# Those are languages that we're currently detecting.
all_categories=['spa','eng']

def get_data(x_file='data/x_train.txt', y_file='data/y_train.txt',
             include_langs=all_categories, exclude_words=[]):
    """
    Read data from files and clean it up, return a list of languages/categories
    that we've got data for and a dictonary with list of unique words for each
    language.

    x_file - in each line contains a sentence in different language
    y_file - in each line contains a language code/label correspoinding
             to sentences in x_file
    include_langs - list of languages codes that you want to include in your
                    data set
    exclude_words - words that should be excluded from this dataset (we use it in
                    our test set to filter out all the words that also included
                    in train set, we just want to have a unique and new words in
                    test set to verify our model accurately)
    """
    languages=[]
    words={}


    y=open(y_file).read().splitlines()
    x=open(x_file, encoding='utf-8').read().splitlines()

    for i, lang in enumerate(y):
        # Include only languages we're interested in
        if include_langs and lang not in include_langs:
            continue
        # If it's a new language add it to the list
        if lang not in languages:
            languages.append(lang)
        if lang not in words:
            words[lang]=[]
        # Get only relevant words for each language
        include_words=[]
        for word in x[i].split():
            # Exclude short words
            if len(word) < 3:
                continue
            # We're interested only in
            # alphabetic data e.g. words.
            if not word.isalpha():
                continue
            # Ignore words on our "black list"
            if word in exclude_words:
                continue
            # Add cleaned up word, make it lowercase.
            include_words.append(unidecode.unidecode(word.lower()))
        words[lang].extend(include_words)

    # Remove duplicate words.
    # This will speed up the processing.
    for l in languages:
        print("Number of words in %s: %d (with duplicates)" % (l, len(words[l])))
        words[l]=list(set(words[l]))

    # We want to make sure that we have the same
    # number of words for all of the langauges.
    # More words for a given language can make our
    # network "know" it better and we don't want that.
    #
    # Find which language has the least amount of words.
    max=0
    for l in languages:
        lw=len(words[l])
        if max == 0:
            max=lw
            continue
        if lw<max:
            max=lw
    print('Maximum number of words we can use:',max)

    # Limit the number of words to the smallest amount
    # we've found above.
    for l in languages:
        wl=len(words[l])
        words[l]=words[l][:max]
        print('Limit words length for %s %d->%d' % (l, wl, len(words[l])))

    return languages, words

def get_data_test(exclude_words):
    """
    Read test data from files and clean it up.

    exclude_words - list of words to filter our from test data,
                    we use it to remove words that are both in
                    train and test datasets to make sure that
                    we test on totally fresh/unseen test dataset.
    """
    return get_data(x_file='data/x_test.txt', y_file='data/y_test.txt', exclude_words=exclude_words)

if __name__ == '__main__':
    from pprint import pprint
    print('Training data:')
    categories, train_words=get_data('data/x_train.txt', 'data/y_train.txt')
    for c in categories:
        pprint((c,train_words[c][:10]))
    print('Test data:')
    categories, test_words=get_data_test(exclude_words=[ train_words[c] for c in all_categories ])
    for c in categories:
        pprint((c,test_words[c][:10]))
