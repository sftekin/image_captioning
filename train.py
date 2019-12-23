import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from batch_generator import BatchGenerator
from models.vgg_lstm import CaptionLSTM
from embedding.embedding import Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, batch_gen, **kwargs):
    net.train()
    net.to(device)

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    opt = optim.Adam(net.parameters(), lr=kwargs['lr'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(kwargs['n_epoch']):
        running_loss = 0
        h = net.init_hidden(batch_size)
        for idx, (im, x_cap, y_cap) in enumerate(batch_gen.generate('train')):

            print('\rtrain:{}'.format(idx), flush=True, end='')

            im, x_cap, y_cap = im.to(device), x_cap.to(device), y_cap.to(device)
            h = tuple([each.data for each in h])

            opt.zero_grad()
            output, h = net(im, x_cap, h)

            loss = criterion(output, y_cap.view(batch_size * seq_length).long())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), kwargs['clip'])
            opt.step()

            running_loss += loss.item()

            if (idx+1) % kwargs['print_every'] == 0:
                print('\n')
                val_loss = evaluate(net, batch_gen, **kwargs)
                print('\n')
                print("Epoch: {}/{}...".format(epoch + 1, kwargs['n_epoch']),
                      "Step: {}...".format(idx),
                      "Loss: {:.4f}...".format(running_loss / idx),
                      "Val Loss: {:.4f}".format(val_loss))
        print('\n')
        print('Creating sample captions')
        sample(net, batch_gen, top_k=5, **kwargs)

    print('Training finished, saving the model')
    model_file = open('vgg_lstm.pkl', 'wb')
    pickle.dump(net, model_file)


def evaluate(net, batch_gen, **kwargs):
    net.eval()

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    criterion = nn.CrossEntropyLoss()

    val_h = net.init_hidden(batch_size)
    val_losses = []
    for idx, (im, x_cap, y_cap) in enumerate(batch_gen.generate('validation')):

        print('\rval:{}'.format(idx), flush=True, end='')

        im, x_cap, y_cap = im.to(device), x_cap.to(device), y_cap.to(device)
        val_h = tuple([each.data for each in val_h])

        output, val_h = net(im, x_cap, val_h)
        val_loss = criterion(output, y_cap.view(batch_size * seq_length))

        val_losses.append(val_loss.item())

    net.train()
    return np.mean(val_losses)


def predict(net, image, x_cap, h=None, top_k=None):
    image = image.unsqueeze(dim=0)

    x_cap = torch.tensor([[x_cap]])

    h = tuple([each.data for each in h])
    out, h = net(image, x_cap,  h)
    p = F.softmax(out, dim=1).data

    if device == 'gpu':
        p = p.cpu()

    p, top_ch = p.topk(top_k)
    top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    word_int = np.random.choice(top_ch, p=p / p.sum())

    return word_int, h


def show_image(img, captions):
    embed = Embedding(dataset_path="./dataset",
                      train_on=False,
                      device=device)

    caption_str = embed.translate(captions)
    plt.figure()
    img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.tight_layout()
    plt.title(caption_str)


def sample(net, batch_gen, top_k=None, **kwargs):
    net.to(device)
    net.eval()

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    im, _, y_cap = next(batch_gen.generate('test'))
    h = net.init_hidden(1)

    x_cap = 1  # x_START_
    captions = []
    for i in range(batch_size):
        caption = []
        for ii in range(seq_length):
            x_cap, h = predict(net, im[i, :], x_cap, h, top_k=top_k)
            caption.append(x_cap)
        captions.append(caption)

    for i in range(batch_size):
        show_image(im[i], captions[i])
    plt.show()

    net.train()
    return captions


if __name__ == '__main__':

    data_params = {
        'embed_dim': 300,
        'vocab_dim': 1004
    }

    model_params = {
        'drop_prob': 0.3,
        'n_layers': 2,
        'n_hidden': 512,
        'transfer_learn': True
    }

    train_params = {
        'n_epoch': 15,
        'clip': 5,
        'lr': 0.01,
        'seq_len': 16,
        'print_every': 500
    }

    batch_creator = BatchGenerator(dataset_path='./dataset',
                                   image_path='./dataset/images/')
    model = CaptionLSTM(model_params=model_params,
                        data_params=data_params)

    train(model, batch_creator, **train_params)
