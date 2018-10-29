import torch

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from prep import get_data, MAX_SENTENCE_LENGTH, EOS_token, SOS_token, UW_token
import random
import os

hidden_size=256

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def sentence_to_idx(lang, sentence):
    """
    Encode sentences to indexes in our Vocabulary object
    for a given langauge.
    """
    out=[]
    for word in sentence.split(' '):
        if word in lang.word2index:
            out.append(lang.word2index[word])
        else:
            out.append(UW_token)
    return out

def sentence_to_tensor(lang, sentence):
    """
    Turn a sentence into a tensor.
    Add EOS_token at the end of the new tensor
    to mark end of the sentence.
    """
    indexes = sentence_to_idx(lang, sentence)
    indexes.append(EOS_token)
    #print('Sentence->Word indexes', sentence, indexes)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def pair_to_tensor(il, ol, pair):
    """
    Turn a pair of sentences into a pair of tensors.
    """
    input_tensor = sentence_to_tensor(il, pair[0])
    output_tensor = sentence_to_tensor(ol, pair[1])
    return (input_tensor, output_tensor)

def train(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_func, max_length=MAX_SENTENCE_LENGTH):
    """
    Encode input_tensor and feed the output to decode the output_tensor.
    """
    encoder_hidden = encoder.initHidden()

    input_length = input_tensor.size(0)
    # Forward pass, process input tensor vi encoder:
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # Prepare input values for decoder, starting with start of the sentence character.
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Make encoder output decoder's input.
    decoder_hidden = encoder_hidden

    output_length = output_tensor.size(0)
    loss = 0
    # Now processing output tensor via decoder:
    for di in range(output_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Return the best guess for current word.
        _, topi = decoder_output.topk(1)
        # Prepare next input from cuurent output
        decoder_input = topi.squeeze().detach()

        # Calculate loss.
        loss += loss_func(decoder_output, output_tensor[di])

        # Stop if it's the end of the sentence.
        if decoder_input.item() == EOS_token:
            break
    # Clean up the "gradients" before
    # propagating changes to our network.
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Accumulate changes.
    loss.backward()

    # Propagate changes.
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_length

def train_all(pairs, encoder, decoder, il, ol, s_epochs, print_every=10):
    """
    Train on a s_epochs random pair of sentences using encoder
    and decoder.

    print_every - show stats on print_every sentence
    """
    loss_total = 0

    # Initialize optimizers for both networks.
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())

    # Get a n_iters random sentences for training.
    training_pairs = [pair_to_tensor(il, ol, random.choice(pairs)) for i in range(s_epochs)]
    loss_func = nn.CrossEntropyLoss()

    # Feed each pair to both of our networks
    for se in range(s_epochs):
        # Get the next pair of sentences to train
        training_pair = training_pairs[se]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # Do the actual training.
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_func)
        loss_total += loss

        if se % print_every == 0:
            loss_avg = loss_total / print_every
            loss_total = 0
            print('%d %d%% %.4f' % (se, se / s_epochs * 100, loss_avg))

def test(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_SENTENCE_LENGTH):
    """
    Generate translation of a sentence using encoder and decoder.
    """
    # This is not training, so we can
    # save up some memory.
    with torch.no_grad():
        # Prepare sentence for translation.
        input_tensor = sentence_to_tensor(input_lang, sentence)
        input_length = input_tensor.size()[0]

        # This is similar to training, but without running
        # .zero_grad(), .backward(), .set()
        encoder_hidden = encoder.initHidden()

        # First encode our sentence using the first network.
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)

        # Prepare data for the second network based on the output
        # of the first one.
        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        # Get the translation using the second network
        # decode it's output.
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Get the best guess.
            topv, topi = decoder_output.data.topk(1)
            # We're done once we get at the end of the sentence.
            if topi.item() == EOS_token:
                break
            else:
                # otherwise just turn the encoded translation (as indexes)
                # back to their respective words.
                decoded_words.append(output_lang.index2word[topi.item()])
            # Preparing the next input for decoder.
            decoder_input = topi.squeeze().detach()

        return decoded_words

def test_random(encoder, decoder,ilang,olang, n=5):
    """
    Randomly get a pair of sentences and compare them
    with our translation.
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('Question in %s: %s' % (ilang.name, pair[0].ljust(20)))
        print('Question in %s: %s' % (olang.name, pair[1].ljust(20)))
        output_words = test(encoder, decoder, pair[0], ilang, olang)
        output_sentence = ' '.join(output_words).strip()
        tick='V' if output_sentence == pair[1] else 'X'
        print('Our guess:%s %s' % (output_sentence.ljust(20), tick))
        print('')

if __name__ == '__main__':
    hidden_size=hidden_size
    # Get maximum of 100 sentences.
    # Remember that in prep.py we get only questions that match secific criteria.
    pairs, input_lang, output_lang=get_data('en','spa', limit=100)
    # Building two GRUs, encoder and decoder.
    encoder = EncoderGRU(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderGRU(hidden_size, output_lang.n_words).to(device)
    print('Training models...')
    train_all(pairs,encoder,decoder,input_lang,output_lang,900,print_every=100)
    print('Saving both models...')
    torch.save(encoder.state_dict(), 'encoder.ckpt')
    torch.save(decoder.state_dict(), 'decoder.ckpt')
    print('Testing with random data...')
    test_random(encoder, decoder, input_lang, output_lang)
