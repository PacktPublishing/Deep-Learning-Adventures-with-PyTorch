"""
Train and test a simple RNN for language detection.

Inspired by
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
import torch
torch.manual_seed(2)

import torch.nn as nn

from prep import get_data, get_data_test, all_categories
import time
import math
import random
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, n_letters, n_categories, hidden_size=56):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_letters + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_letters + hidden_size, n_categories)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def wtotensor(word):
    """
    Encode a word as a tensor using a standard alphabet (defined in all_letters)

    For example:
    Give our alphabet:
    abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'-

    Each lettter has a uniqur position:
    0 -> a
    1 -> b
    etc.
    15 -> o

    So, if we want to encode the word 'oro' we will encode each letter
    by including 1 in it's position and left the other positions as 0:

    oro->
                           o is in 15th position in the alphabet--V
tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
r in 18th->1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0.]],
                 and again o is in 15th position in the alphabet--V
        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]]])

    """
    tensor = torch.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def random_value(d):
    """
    Get the random value from dictonary d.

    We use this function both to get the random
    language/category as well as a word.
    """
    return d[random.randint(0, len(d) - 1)]

def get_tensorw(all_categories, words, category=None, word=None):
    """
    Get a random category and word, return tensors for both.
    If category and word is specified just turn them into tensors
    and return.
    """
    if category is None and word is None:
        category = random_value(all_categories)
        word = random_value(words[category])
    category_tensor = torch.LongTensor([all_categories.index(category)])
    word_tensor = wtotensor(word)
    return category, word, category_tensor, word_tensor

def get_category(output, categories):
    """
    Return the most probable category/language
    from output tensor.
    """
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return categories[category_i], category_i

def train(rnn, optimizer, loss_function, w_epochs, categories, words):
    """
    Train rmm model using optimizer, loss_function on w_epochs words
    based on categories with words.
    """
    print('Starting training...')
    current_loss=0
    wordi=0
    stats_total={}.fromkeys(categories, 0)
    for w_epoch in range(1, w_epochs + 1):
        wordi+=1
        # Get random data for training.
        category, word, category_tensor, word_tensor = get_tensorw(categories, words)
        stats_total[category]+=1
        # We need to initalize our
        # hidden variable first.
        hidden = rnn.initHidden()

        # Forward pass: predict a language for each character
        # in a word.
        for i in range(word_tensor.size()[0]):
            output, hidden = rnn(word_tensor[i], hidden)

        # Calculate the difference between
        # what we've predicted and what we should
        # predict.
        loss = loss_function(output, category_tensor)

        # Because changes('gradients') are accumulated
        # from one iteration to another we need to
        # clean up the last ones, so we can propagate
        # the ones from this iteration.
        # Note: always call it before
        # loss.backward() and optimizer.step()
        optimizer.zero_grad()

        # Backward pass: accumulate changes('gradients')
        # that we've learned about in this iteration.
        loss.backward()

        # Backward pass: propagate changes trough the network.
        optimizer.step()

        loss=loss.data.item()
        current_loss += loss
        # Print progress every now and then.
        if wordi % 1000 == 0:
            guess, _ = get_category(output, categories)
            if guess == category:
                msg = 'V'
            else:
                msg='X (%s)' % category
            print('%d %d%% %s %s %s %f' % (w_epoch, w_epoch / w_epochs * 100,
            word.ljust(20), guess, msg.ljust(8), current_loss / 1000))
            current_loss=0.0
    print('Fnished training on %d words' % wordi)
    for c in categories:
        print('Trained on %d words for %s' % (stats_total[c], c))

def test(rnn, optimizer, categories, test_words):
    """
    Test data on all test dataset, calculate how
    much images have been classified correctly.

    We testing the model in a similar way that we do
    training, but we're going trough test set word by word
    (not randomly like in training).

    We're counting the total number of words for each language
    and also a number of words that were detected correctly.
    """
    stats_correct={}.fromkeys(categories, 0)
    stats_total={}.fromkeys(categories, 0)
    print('Starting testing...')
    with torch.no_grad():
        for cat in categories:
            for w in test_words[cat]:
                _, _, category_tensor, word_tensor = get_tensorw(categories, test_words, cat, w)
                hidden = rnn.initHidden()

                for i in range(word_tensor.size()[0]):
                    output, hidden = rnn(word_tensor[i], hidden)

                guess, _ = get_category(output, categories)
                stats_total[cat]+=1
                if (guess == cat):
                    stats_correct[cat]+=1
        for c in categories:
            print('Test accuracy for %s on %d (%d correct) words:%d %%' % (c, stats_total[c], stats_correct[c], 100 *  stats_correct[c] / stats_total[c]))

if __name__ == '__main__':
    # Initialize our language detector
    rnn = RNN(n_letters, n_categories)
    # Initialize optimizer
    optimizer = torch.optim.Adam(rnn.parameters())
    # Initialize our loss function
    loss_function = nn.CrossEntropyLoss()
    # Get training data
    print('Getting training data...')
    categories, train_words=get_data()
    # Train using 10000 words choosen randomly for
    # each language, in general we get around 50% words
    # for each language.
    train(rnn, optimizer, loss_function, 10000, categories, train_words)
    # Get test data, don't include words from training set.
    print('Getting test data...')
    test_categories, test_words=get_data_test(
                exclude_words=[ train_words[c] for c in all_categories ])
    # Test our model on totally fresh and unique list of words.
    test(rnn, optimizer, test_categories, test_words)
    # Save our model,so we can use it for detection.
    torch.save(rnn.state_dict(), 'model.ckpt')
