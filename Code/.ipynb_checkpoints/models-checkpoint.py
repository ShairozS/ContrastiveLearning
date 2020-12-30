from torch import nn



class SiameseModel(nn.Module):
    
    
    def __init__(self, in_channels=3):
        
        super(SiameseModel, self).__init__()
        
        self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=10),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
                nn.MaxPool2d(kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
                nn.ReLU())
        
        
        self.fc = nn.Sequential(nn.Linear(in_features=9216, out_features=4096), 
                                nn.Linear(in_features=4096, out_features=1),
                                nn.Sigmoid())
