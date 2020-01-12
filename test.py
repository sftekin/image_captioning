import torch
import numpy as np

from transformers.bleu import compute_bleu
from train_helper import predict, translate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, batch_gen, top_k, **kwargs):
    net.eval()
    net.to(device)

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    translate_caps = []
    referance_caps = []
    for idx, (im, x_cap, y_cap) in enumerate(batch_gen.generate('test')):

        print('\rtest:{}'.format(idx), flush=True, end='')

        im, x_cap, y_cap = im.to(device), x_cap.to(device), y_cap.to(device)

        word_int = 1  # x_START_
        for i in range(batch_size):
            caption = []
            h = None
            for ii in range(seq_length):
                word_int, h = predict(net, im[i, :], word_int, h, top_k=top_k)
                caption.append(word_int)
            caption = translate(caption, net.embed_layer.int2word)
            translate_caps.append(caption.split())

        for i in range(batch_size):
            y_trimed = trim_empty_rows(y_cap[i].numpy())
            y_str = [translate(y_trimed[ii], net.embed_layer.int2word).split() for ii in range(y_trimed.shape[0])]
            referance_caps.append(y_str)

        print('\n')
        for i in range(1, 5):
            bleu, geo_mean, bp = compute_bleu(referance_caps, translate_caps, max_order=i)
            print('BLEU: {}, Geometric_mean: {}, BP:{}'.format(bleu, geo_mean, bp))

    print('\n')
    for i in range(1, 5):
        bleu, geo_mean, bp = compute_bleu(referance_caps, translate_caps, max_order=i)
        print('BLEU: {}, Geometric_mean: {}, BP:{}'.format(bleu, geo_mean, bp))


def trim_empty_rows(y_cap):
    return y_cap[~np.all(y_cap == 0, axis=1)]

