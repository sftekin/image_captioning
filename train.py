import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from batch_generator import BatchGenerator
from models.vgg_lstm import CaptionLSTM
from load_data import LoadData
from embedding import Embedding

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

            # weight = torch.ones(1004)
            # weight[[0, 1, 2, 3]] = 0
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
        model_file = open('vgg_lstm.pkl', 'wb')
        pickle.dump(net, model_file)

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

    x_cap = torch.tensor([[x_cap]]).to(device)

    # h = tuple([each.data for each in h])
    out, h = net(image, x_cap,  h)
    p = F.softmax(out, dim=1).data

    if torch.cuda.is_available():
        p = p.cpu()

    p, top_ch = p.topk(top_k)
    top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    word_int = np.random.choice(top_ch, p=p / p.sum())

    return word_int, h


def sample(model, batch_gen, top_k=None, **kwargs):
    model.to(device)
    model.eval()

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    im, _, y_cap = next(batch_gen.generate('validation'))
    im, y_cap = im.to(device), y_cap.to(device)

    x_cap = 1  # x_START_
    captions = []
    for i in range(batch_size):
        caption = []
        h = model.init_hidden(1)
        for ii in range(seq_length):
            x_cap, h = predict(model, im[i, :], x_cap, h, top_k=top_k)
            caption.append(x_cap)
        captions.append(caption)

    for i in range(batch_size):
        show_image(im[i], captions[i], model)
    plt.show()

    model.train()
    return captions


def show_image(img, captions, model):
    if torch.cuda.is_available():
        img = img.cpu()

    caption_str = translate(captions, model.embed_layer.int2word)
    plt.figure()
    img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.tight_layout()
    plt.title(caption_str)


def translate(captions, int2word):
    caption_str = ' '.join([int2word[cap] for cap in captions])
    return caption_str


if __name__ == '__main__':
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

    print('Loading data...')
    data = LoadData(dataset_path='dataset',
                    images_path='dataset/images/')

    print('Creating Batch Generator...')
    batch_creator = BatchGenerator(data_dict=data.data_dict,
                                   captions_int=data.captions_int,
                                   image_addr=data.image_addr)

    print('Creating Models...')
    caption_model = CaptionLSTM(model_params=model_params,
                        int2word=data.int2word)

    print('Starting training...')
    train(caption_model, batch_creator, **train_params)
    # model_file = open('vgg_lstm.pkl', 'rb')
    # model = pickle.load(model_file)
    #
    # sample(model, batch_creator, top_k=10, seq_len=16)

