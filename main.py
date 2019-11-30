import os

from batch_generator import BatchGenerator


def captions_to_words(int2word, captions):
    sentence = []
    for caption in captions:
        sentence.append([int2word[word] for word in caption])
    return sentence


def main():
    dataset_path = 'dataset'
    images_path = 'images'
    local_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_path = os.path.join(local_dir, dataset_path)
    images_path = os.path.join(dataset_path, images_path)

    batch_gen = BatchGenerator(dataset_path, images_path)
    for batch_idx, (image, caption) in enumerate(batch_gen.generate('test', batch_format='embedding')):
        print(image.shape)  # (b, s, height, width)
        print(caption.shape)  # (b, s, embed_dim)


if __name__ == '__main__':
    main()
