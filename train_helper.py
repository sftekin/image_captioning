import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(net, image, x_cap, h=None, top_k=None):
    image = image.unsqueeze(dim=0)

    x_cap = torch.tensor([[x_cap]]).to(device)

    out, h = net(image, x_cap, sentence_len=1, hidden=h)
    p = F.softmax(out, dim=1).data

    if torch.cuda.is_available():
        p = p.cpu()

    p, top_ch = p.topk(top_k)
    top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()
    word_int = np.random.choice(top_ch, p=p / p.sum())

    return word_int, h


def sample(net, batch_gen, top_k=None, **kwargs):
    net.to(device)
    net.eval()

    batch_size = batch_gen.batch_size
    seq_length = kwargs['seq_len']

    im, _, y_cap = next(batch_gen.generate('validation'))
    im, y_cap = im.to(device), y_cap.to(device)

    x_cap = 1  # x_START_
    captions = []
    for i in range(batch_size):
        print('\rsample:{}'.format(i), flush=True, end='')
        caption = []
        h = None
        for ii in range(seq_length):
            x_cap, h = predict(net, im[i, :], x_cap, h, top_k=top_k)
            caption.append(x_cap)
        captions.append(caption)

    print('\n')
    for i in range(batch_size):
        caption_str = translate(captions[i], net.embed_layer.int2word)
        print('Caption {}: {}'.format(str(i), caption_str))
        if kwargs['show_image']:
            show_image(im[i], caption_str)
    plt.show()

    net.train()
    return captions


def show_image(img, caption_str):
    if torch.cuda.is_available():
        img = img.cpu()

    plt.figure()
    img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.tight_layout()
    plt.title(caption_str)


def translate(captions, int2word, remove_unk=False):
    caption_str = ' '.join([int2word[cap] for cap in captions])
    caption_str = caption_str.replace('x_END_', '').replace('x_NULL_', '')
    caption_str = ' '.join(caption_str.split())
    if remove_unk:
        caption_str = caption_str.replace('x_UNK_', '')
        caption_str = ' '.join(caption_str.split())
    return caption_str


def calc_class_weights(captions_int):
    counts = collections.Counter(captions_int.flatten())
    counts_array = np.array(list(counts.values()))

    # calculating idf
    word_count = len(captions_int.flatten())
    counts_array = np.log(word_count / counts_array)
    counts_tensor = torch.from_numpy(counts_array).float().to(device)

    return counts_tensor
