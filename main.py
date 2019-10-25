import os
import matplotlib.pyplot as plt

from skimage import io
from torch.utils.data import DataLoader

from load import LoadData
from data import ImageDataset


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

    caption_path = os.path.join(dataset_path, 'captions.csv')
    caption_word_path = os.path.join(dataset_path, 'captions_words.csv')
    im_addr_path = os.path.join(dataset_path, 'imid.csv')
    dataset_h5_file_path = os.path.join(dataset_path, 'eee443_project_dataset_train.h5')
    image_path_names = [os.path.join(images_path, f) for f in sorted(os.listdir(images_path))]

    loaded_data = LoadData(dataset_h5_file_path, caption_path,
                           caption_word_path, im_addr_path)

    sample_dataset = ImageDataset(image_path_names,
                                  loaded_data.captions,
                                  loaded_data.im_addr)

    sample_img, caption = sample_dataset[4]
    sentences = captions_to_words(loaded_data.int2word, caption)
    for sentence in sentences:
        print(sentence)

    io.imshow(sample_img)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # sample_dataloader = DataLoader(sample_dataset,
    #                                batch_size=64,
    #                                shuffle=True,
    #                                num_workers=4)
    #
    # for batch_idx, (img, caption) in enumerate(sample_dataloader):
    #     a, b = img, caption


if __name__ == '__main__':
    main()
