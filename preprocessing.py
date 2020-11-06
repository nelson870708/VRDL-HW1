import os

import pandas as pd
from PIL import Image
from sklearn.utils import shuffle

in_dir = './data/training_data'
out_train_dir = './data/train/'
out_val_dir = './data/val/'
train = pd.read_csv('./data/training_labels.csv')
train = shuffle(train)

if not os.path.isdir('./data/train'):
    os.mkdir('./data/train')
if not os.path.isdir('./data/val'):
    os.mkdir('./data/val')

for idx in range(train.shape[0]):
    data_path = in_dir + '/' + str(train['id'][idx]).zfill(6) + '.jpg'
    im = Image.open(data_path)
    class_name = train['label'][idx].translate({ord('/'): '_'})
    if not os.path.isdir(out_train_dir + class_name) or not os.path.isdir(out_val_dir + class_name):
        os.mkdir(out_train_dir + class_name)
        os.mkdir(out_val_dir + class_name)
    if idx < train.shape[0] * 0.8:
        im.save(out_train_dir + class_name + '/' + str(train['id'][idx]).zfill(6) + '.jpg')
    else:
        im.save(out_val_dir + class_name + '/' + str(train['id'][idx]).zfill(6) + '.jpg')
    im.close()

