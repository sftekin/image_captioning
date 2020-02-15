# Image Captioning with <br/> Transfer Learning, Attention, Pretrained Embeddings, and Teacher Forching Techniques

## Installing

### Python Dependencies

*Using Docker*


You can create docker image using Dockerfile
```
    $ cd Docker
    $ docker build -t im_caption:v1 .
```
then you can create docker container

```
    $ docker run -it --rm -v `pwd`:/workspace im_caption:v1
```
then you can run your `python` commands from there 

*Using Conda*

- First download pytorch and torchvision from https://pytorch.org using conda
- Then, you can download required libraries using `pip` inside `Docker/Dockerfile`

 

### Data Set
Selected dataset was flicker 30k in format of `.h5`. (Right now I am trying to provide a download link)

- This file includes urls for the images. To download images from those urls run `data_extractor.py`

- dataset includes image ids, integer captions, word captions. You need to create `.csv` files

    - These files should be created `captions_words.csv`, `captions.csv`, `imid.csv` and `word2int.csv`
    - (Right now I will try to provide the code for creation of `.csvs`)

## Methodology

### Transfer Learning
For CNN architecture on image feature extraction pretrained VGG16 is used,

```
self.vgg = models.vgg16(pretrained=self.pre_train)
``` 

Until 15th epoch gradients of VGG16 is freezed. After Epoch 15 last layers activated


### Attention
Attention model is implementation of [Show, Attend, and Tell](https://arxiv.org/abs/1502.03044) paper

In the implementation I get help from this [repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) 

### Pretrained Embeddings
   
I have downloaded the [Fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) trained embeddings for English

In embedding layer `word2vec.py` transformer is called for every word in the dictionary. 

Obtained vectors create the weights of embedding layer.

### Teacher Forcing
   
In `dataset.py` you can see how the target captions are created.
   
```
# Target captions are one step forward of train captions
target_captions = np.roll(train_captions, -1) 
```  

Target captions are one step forward of training captions. 

Thus network tries to predict next word from previous word given for training.


## Running

You can run


