import torch
import torch.nn as nn
import torchvision.transforms as transforms
VGG_types = {
    'VGG11' : [64, 128, 'M', 256, 512, 'M'],
}

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, init_weights=True, model='VGG11'):
        super(Net, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_layers = self.create_conv_layers(VGG_types[model])
        
        self.fc_layers = nn.Sequential(
            # 28 / 2 ** 2
            nn.Linear(512 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(1000, num_classes)
            )
        if init_weights == True:
            self._initialize_weights()


    def forward(self, x):
#       print(x.shape)
        x = transforms.Grayscale(num_output_channels = 1)(x)
        #print(x.shape)
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
            # modules -> Sequential 모든 layer
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # he_initialization
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_out',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.normal_(m.bias, 0)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:  # conv layer
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                # channel num --> next input
                in_channels = x

            elif x == 'M':  # maxpooling
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)