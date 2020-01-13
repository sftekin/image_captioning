import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim


from train_helper import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, batch_gen, weights, **kwargs):
    net.train()
    net.to(device)

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    opt = optim.Adam(net.parameters(), lr=kwargs['lr'])
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loss_list = []
    val_loss_list = []
    for epoch in range(kwargs['n_epoch']):
        running_loss = 0
        for idx, (im, x_cap, y_cap) in enumerate(batch_gen.generate('train')):

            print('\rtrain:{}'.format(idx), flush=True, end='')

            im, x_cap, y_cap = im.to(device), x_cap.to(device), y_cap.to(device)

            opt.zero_grad()
            output, _ = net(im, x_cap)

            loss = criterion(output, y_cap.view(batch_size * seq_length).long())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), kwargs['clip'])
            opt.step()

            running_loss += loss.item()

            if (idx+1) % kwargs['eval_every'] == 0:
                print('\n')
                val_loss = evaluate(net, batch_gen, weights, **kwargs)
                print("\nEpoch: {}/{}...".format(epoch + 1, kwargs['n_epoch']),
                      "Step: {}...".format(idx),
                      "Loss: {:.4f}...".format(running_loss / idx),
                      "Val Loss: {:.4f}".format(val_loss))

        # After 15 epochs open last parts of conv model
        if epoch > 15:
            net.fine_tune()

        print('Creating sample captions')
        sample(net, batch_gen, top_k=5, **kwargs)
        print('\n')

        train_loss_list.append(running_loss / idx)
        val_loss_list.append(val_loss)

        loss_file = open('losses.pkl', 'wb')
        model_file = open('vgg_lstm.pkl', 'wb')
        pickle.dump([train_loss_list, val_loss_list], loss_file)
        pickle.dump(net, model_file)

    print('Training finished, saving the model')
    model_file = open('vgg_lstm.pkl', 'wb')
    pickle.dump(net, model_file)


def evaluate(net, batch_gen, weights, **kwargs):
    net.eval()

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    criterion = nn.CrossEntropyLoss(weight=weights)

    val_losses = []
    for idx, (im, x_cap, y_cap) in enumerate(batch_gen.generate('validation')):

        print('\rval:{}'.format(idx), flush=True, end='')

        im, x_cap, y_cap = im.to(device), x_cap.to(device), y_cap.to(device)

        output, _ = net(im, x_cap)
        val_loss = criterion(output, y_cap.view(batch_size * seq_length))

        val_losses.append(val_loss.item())

    net.train()
    return np.mean(val_losses)

