"""
Use already trained GRUS to translate
short text from one language to the other.
"""
import torch
from prep import get_data
from train import device, hidden_size, EncoderGRU, DecoderGRU, test
import sys
import os

def dummy(*args, **kwargs):
    """
    This is the place where all
    the messages got lost...
    """
    pass

if  __name__ == '__main__':
    # We could save both input_lang and output_lang
    # on training so we didn't have to load data from scratch.
    # Here, we don't have a lot of data, so it's fine to do that.
    _, input_lang, output_lang=get_data('en','spa', limit=100, log=dummy)
    # Prepare GRUS
    encoder = EncoderGRU(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderGRU(hidden_size, output_lang.n_words).to(device)
    # Loading models...
    if os.path.exists('encoder.ckpt') and os.path.exists('decoder.ckpt'):
        print('Using trained models...')
        encoder.load_state_dict(torch.load('encoder.ckpt'))
        decoder.load_state_dict(torch.load('decoder.ckpt'))
    # Doing the translation...
    guess=test(encoder, decoder,  sys.argv[1], input_lang, output_lang)
    print(sys.argv[1],'-> '+' '.join(guess))
