from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader


class BatchGenerator:
    def __init__(self, data_dict, captions_int, image_addr, **kwargs):
        self.data_dict = data_dict
        self.captions_int = captions_int
        self.image_addr = image_addr

        self.batch_size = kwargs.get('batch_size', 16)
        self.num_works = kwargs.get('num_works', 4)
        self.shuffle = kwargs.get('shuffle', True)
        self.use_transform = kwargs.get('use_transform', True)
        self.input_size = kwargs.get("input_size", (224, 224))

        self.dataset_dict, self.dataloader_dict = self.__create_data()

    def generate(self, data_type):
        """
        :param data_type: can be 'test', 'train' and 'validation'
        :return: img tensor, label numpy_array
        """
        selected_loader = self.dataloader_dict[data_type]
        yield from selected_loader

    def __create_data(self):
        if self.use_transform:
            im_transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            im_transform = None

        im_dataset = {}
        for i in ['test', 'train', 'validation']:
            return_all = True if i == 'test' else False
            im_dataset[i] = ImageDataset(image_path_names=self.data_dict[i],
                                         captions_int=self.captions_int,
                                         im_addr=self.image_addr,
                                         transformer=im_transform,
                                         return_all=return_all)

        im_loader = {}
        for i in ['test', 'train', 'validation']:
            im_loader[i] = DataLoader(im_dataset[i],
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      num_workers=self.num_works,
                                      drop_last=True)
        return im_dataset, im_loader
