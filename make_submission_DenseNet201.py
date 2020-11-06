import os
import pickle

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from DenseNet201 import SquarePad

n_classes = 196
input_size = 224
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet201(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, n_classes)
model.load_state_dict(torch.load('./model/model_DenseNet201'))
model.to(device)
model.eval()

data_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_obj(dic_name):
    with open(dic_name + '.pkl', 'rb') as f:
        return pickle.load(f)


class_dict = load_obj('dict_DenseNet201')
inv_class_dict = {v: k for k, v in class_dict.items()}

pred_list = []
id_list = []
for root, dirs, files in os.walk("./data/testing_data", topdown=False):
    for name in files:
        img = Image.open(os.path.join(root, name))
        img_rgb = img.convert('RGB')
        x_test = data_transforms(img_rgb).to(device)
        x_test.unsqueeze_(0)  # Add batch dimension
        output = model(x_test)
        pred = torch.argmax(output, 1)
        if pred.cpu().numpy()[0] > 195:
            class_pred = inv_class_dict[0]
        else:
            class_pred = inv_class_dict[pred.cpu().numpy()[0]]
        id_list.append(name)
        pred_list.append(class_pred.replace('_', '/'))
id_list = [img_id.split('.')[0] for img_id in id_list]

dic = {
    'id': id_list,
    'label': pred_list
}
df = pd.DataFrame(dic)
df.to_csv('submission_DenseNet201.csv', index=False)
