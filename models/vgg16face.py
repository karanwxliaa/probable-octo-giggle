import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

class vgg16_pt(nn.Module):
    def __init__(self, num_features=21):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        # Convolutional Layers
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        # Fully Connected Layers for VGG16
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 256)
        
        # Custom Fully Connected Layers for tasks
        self.last = nn.Linear(256, num_features)
        self.load_weights()
        
    def load_weights(self, path="./models/vgg_face_weights/VGG_FACE.t7"):
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, f"conv_{block}_{counter}")
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)
                else:
                    if block in [6, 7]:
                        self_layer = getattr(self, "fc%d" % (block))
                        self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)
                        self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)
                    elif block == 8:
                        break
                    block += 1
                    
    def forward(self, x):
        # Follow the structure of VGG16 forward pass
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc8(x))
        x = F.dropout(x, 0.5, self.training)
        out = {}
        try:
            out['FER'] = self.last['FER'](x)
        except:
            pass 

        try:
            out['AV'] = self.last['AV'](x)
        except:
            pass 

        try:
            out['AU'] = self.last['AU'](x)
        except:
            pass 

        return out

def custom_vgg16_face():
    return vgg16_pt()

