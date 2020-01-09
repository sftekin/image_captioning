import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


from batch_generator import BatchGenerator
from embedding import Embedding


def show_image(img, captions):
    if torch.cuda.is_available():
        img = img.cpu()
    embed = Embedding(dataset_path="./dataset",
                      train_on=False,
                      device='cpu')

    caption_str = embed.translate(captions)
    plt.figure()
    img = (img.permute(1, 2, 0) - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.tight_layout()
    plt.title(caption_str)


if __name__ == '__main__':
    model_file = open('vgg_lstm_cpu.pkl', 'rb')
    model = pickle.load(model_file)

    model.eval()

    batch_gen = BatchGenerator(dataset_path='./dataset',
                               image_path='./dataset/images/',
                               batch_size=1)

    im, _, _ = next(batch_gen.generate('test'))

    word_int = 1
    h = model.init_hidden(1)

    caption = []
    for i in range(16):
        word_int = torch.tensor([[word_int]])
        h = tuple([each.data for each in h])
        out, h = model(im, word_int, h)
        p = F.softmax(out, dim=1).data

        p, top_ch = p.topk(3)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        word_int = np.random.choice(top_ch, p=p / p.sum())
        caption.append(word_int)

    # print(caption)
    show_image(im.squeeze(), caption)
    plt.show()
    # p = p.numpy().squeeze()
    # word_int = np.random.choice(top_ch, p=p / p.sum())

