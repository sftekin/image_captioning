from torch.utils.data import Dataset
from skimage import io


class ImageDataset(Dataset):
    
    def __init__(self, image_path_names, captions, im_addr, transformer=None):
        
        self.image_path_names = image_path_names
        self.captions = captions
        self.im_addr = im_addr
        self.transformer = transformer

    def __len__(self):
        return len(self.image_path_names)
    
    def __getitem__(self, idx):
        
        im_path = self.image_path_names[idx]
        image_name = im_path.split('/')[-1]
        image_id = int(image_name.split('.')[0]) + 1

        caption_idx = self.im_addr[self.im_addr['im_addr'] == image_id].index
        target_captions = self.captions.iloc[caption_idx]

        image = io.imread(im_path)
        if self.transformer is not None:
            image = self.transformer(image)
        
        return image, target_captions.values
