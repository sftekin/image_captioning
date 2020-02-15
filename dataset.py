import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, image_path_names, captions_int,
                 im_addr, transformer, return_all=False):
        self.image_path_names = image_path_names
        self.captions_int = captions_int
        self.im_addr = im_addr
        self.transformer = transformer
        self.return_all = return_all

    def __len__(self):
        return len(self.image_path_names)
    
    def __getitem__(self, idx):

        im_path = self.image_path_names[idx]
        image_name = im_path.split('/')[-1]
        image_id = int(image_name.split('.')[0]) + 1

        caption_idx = self.im_addr[self.im_addr['im_addr'] == image_id].index
        if self.return_all:
            # not all images have 5 captions
            train_captions = np.zeros((6, 17))
            select_cap = self.captions_int.iloc[caption_idx].values
            train_captions[:select_cap.shape[0], :select_cap.shape[1]] = select_cap
        else:
            caption_idx = np.random.choice(caption_idx.values)
            train_captions = self.captions_int.iloc[caption_idx].values

        # Target captions are one step forward of train captions
        target_captions = np.roll(train_captions, -1)
        if self.return_all:
            target_captions[:, -1] = 0
        else:
            target_captions[-1] = 0

        image = Image.open(im_path)
        image = image.convert('RGB')
        if self.transformer is not None:
            image = self.transformer(image)

        return image, train_captions, target_captions
