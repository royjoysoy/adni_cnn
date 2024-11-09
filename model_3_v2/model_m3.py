import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom3DCNN(nn.Module):
    def __init__(self):
        super(Custom3DCNN, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Convolution Block 2
        self.conv2 = nn.Conv3d(64, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=3)

        # Convolution Block 3
        self.conv3 = nn.Conv3d(64, 64, kernel_size=7, stride=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu3 = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 3 * 3 * 3, 864)
      

    def forward(self, x):
        # Convolution Blocks
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # Fully Connected Layers
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc1(out)
  
        # Multi-class Output
        return out

class SimCLR3DCNN(nn.Module):
    def __init__(self, feature_dim=256, num_classes=3):  # Increased feature_dim
        super(SimCLR3DCNN, self).__init__()
        self.f = Custom3DCNN()
        self.f.fc3 = nn.Identity()
        self.f.softmax = nn.Identity()
        self.adaptive_layer = None
        
        # Enhanced projection head with larger dimensions
        self.g = nn.Sequential(
            nn.Linear(1728, 2048, bias=False),  # Larger intermediate dimension
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim, bias=True)
        )
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.classifier = None
        
        # Enable gradient checkpointing for memory efficiency
        self.use_checkpointing = True

    def forward(self, x, return_features=False):
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.f, x)
        else:
            x = self.f(x)
            
        x = x.view(x.size(0), -1)
        
        if self.adaptive_layer is None:
            self.adaptive_layer = nn.Linear(x.shape[1], 1728).to(x.device)
            # Initialize with Kaiming initialization
            nn.init.kaiming_normal_(self.adaptive_layer.weight)
            if self.adaptive_layer.bias is not None:
                nn.init.zeros_(self.adaptive_layer.bias)
                
        x = self.adaptive_layer(x)
        feature = x
        
        if return_features:
            return feature
        
        if self.use_checkpointing and self.training:
            out = torch.utils.checkpoint.checkpoint(self.g, feature)
        else:
            out = self.g(feature)
            
        return F.normalize(out, dim=-1)
            

    def get_features(self, x):
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.f, x)
        else:
            x = self.f(x)
        x = x.view(x.size(0), -1)
        if self.adaptive_layer is None:
            self.adaptive_layer = nn.Linear(x.shape[1], 1728).to(x.device)
            nn.init.kaiming_normal_(self.adaptive_layer.weight)
            if self.adaptive_layer.bias is not None:
                nn.init.zeros_(self.adaptive_layer.bias)
        x = self.adaptive_layer(x)
        return x

    def freeze_encoder(self):
        for param in self.f.parameters():
            param.requires_grad = False
        if self.adaptive_layer is not None:
            for param in self.adaptive_layer.parameters():
                param.requires_grad = False

    def add_classifier(self):
        if self.classifier is None:
            self.classifier = nn.Linear(1728, self.num_classes).to(next(self.parameters()).device)

    def classify(self, x):
        if self.classifier is None:
            raise RuntimeError("Classifier has not been added to the model yet. Call add_classifier() first.")
        feature = self.get_features(x)
        return self.classifier(feature)