import os

from batch_generator import BatchGenerator


def main():
    dataset_path = 'dataset'
    images_path = 'images'
    local_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_path = os.path.join(local_dir, dataset_path)
    images_path = os.path.join(dataset_path, images_path)

    batch_gen = BatchGenerator(dataset_path, images_path)

    for idx, (im, cap) in enumerate(batch_gen.generate('train', batch_format='embedding')):
        print(im.shape)
        print(cap.shape)


if __name__ == '__main__':
    main()
