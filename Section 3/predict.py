"""
Detect one of two languages in a word using our model.
"""
import sys
import torch
from prep import all_categories
from train import n_letters, n_categories, wtotensor
from train import RNN
import unidecode

def predict(rnn, word, n_predictions=n_categories):
    """
    Classify a word to different languages.
    """
    # Turn word into tensor that we can use
    # for language prediction.
    word_tensor=wtotensor(word)

    # Initialize our hidden variable.
    hidden = rnn.initHidden()

    # Use network to detect language for
    # each character.
    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)

    # Get the output in numpy array
    # and figure out the languages
    # that we've detected.
    out=output.detach().numpy()
    aao=[]
    for i, o in enumerate(out[0]):
        iv='?'
        try:
            iv=all_categories[i]
        except KeyError:
            pass
        aao.append((i, o, iv))
    # Just sort our array to show most probable languages first.
    aao.sort(key=lambda x: x[1], reverse=True)
    print('Most probable language for this word is', aao[0][2])
    msg='Energies for each language: %s' % ','.join([ (aao[ci][2]+'(%f)' % aao[ci][1]) for ci in range(2) ])
    print(msg)

if __name__ == '__main__':
    rnn = RNN(n_letters, n_categories)
    rnn.load_state_dict(torch.load('model.ckpt'))
    rnn.eval()
    predict(rnn, unidecode.unidecode(sys.argv[1]))
