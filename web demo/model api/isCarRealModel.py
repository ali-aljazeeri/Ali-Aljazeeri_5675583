import torch
import torch.nn as nn
from torchvision import models, transforms


class IsCarRealModel_resnet152(nn.Module):
    def __init__(self, pretrained=True):
        super(IsCarRealModel_resnet152, self).__init__()
        self.resnet = models.resnet152(pretrained=pretrained)

        # replace avgpool for flexibility
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # custom FC head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    

class IsCarRealModel_resnet101(nn.Module):
    def __init__(self, pretrained=True):
        super(IsCarRealModel_resnet101, self).__init__()
        self.resnet = models.resnet101(pretrained=pretrained)

        # replace avgpool for flexibility
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # custom FC head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    
class IsCarRealModel_resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(IsCarRealModel_resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace avgpool for flexibility
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom FC head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    

class IsCarRealModel_resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(IsCarRealModel_resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)

        # Custom FC head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    
class IsCarRealModel_vgg19(nn.Module):
    def __init__(self, pretrained=True, freeze_features=True):
        super(IsCarRealModel_vgg19, self).__init__()

        # Load pretrained VGG16
        self.vgg = models.vgg19(pretrained=pretrained)

        # Optionally freeze early layers
        if freeze_features:
          # Freeze up to conv3_3
            for param in self.vgg.features[:20]:
                param.requires_grad = False

        # Feature extractor
        self.feature_extractor = nn.Sequential(*list(self.vgg.features.children()))

        # Global average pooling: output size [B, 512, 1, 1] -> [B, 512]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class IsCarRealModel_vgg16(nn.Module):
    def __init__(self, pretrained=True, freeze_features=True):
        super(IsCarRealModel_vgg16, self).__init__()

        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)

        # Optionally freeze early layers
        if freeze_features:
          # Freeze up to conv3_3
            for param in self.vgg.features[:20]:
                param.requires_grad = False

        # Feature extractor
        self.feature_extractor = nn.Sequential(*list(self.vgg.features.children()))

        # Global average pooling: output size [B, 512, 1, 1] -> [B, 512]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Compact classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class IsCarRealModel_mobilenet_v2(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(IsCarRealModel_mobilenet_v2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        if freeze:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False

        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Binary logit
        )

    def forward(self, x):
        return self.mobilenet(x)
    

class IsCarRealModel_mobilenet_v3_large(nn.Module):
    def __init__(self, pretrained=True):
        super(IsCarRealModel_mobilenet_v3_large, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(pretrained=pretrained)

        num_ftrs = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.mobilenet(x)
