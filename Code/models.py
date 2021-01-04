from torch import nn



class ContrastiveModel(nn.Module):
    '''
    A model to produce embedding vectors given an input batched tensor of images
    
    * Note: Input images must be batched tensors of shape (B, in_channels, 105, 105)
    
    Args:
      in_channels (int) : The number of channels in the input image (default=3)
      emb_size (int): The length of the produced embeddings (default=4096)
    '''
    
    def __init__(self, in_channels=3, emb_size=4096):
        
        super(ContrastiveModel, self).__init__()
        
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
        
        
        self.projector = nn.Linear(in_features=9216, out_features=emb_size)
        
        
    def forward(self, x):
        
        b, c, h, w = x.shape
        assert (h==w and h==105), "Inputs must be shaped (B, in_channels, 105, 105)"

        emb = self.backbone(x); b,c,h,w = emb.shape
        emb = emb.view(b, c*h*w)
        emb = self.projector(emb)

        return(emb)
    
    
    def predict(self, x, y, threshold):
        b, c, h, w = x.shape
        assert (h==w and h==105), "Inputs must be shaped (B, in_channels, 105, 105)"
        
        self.model.eval()
        
        emb1 = self.forward(x)
        emb2 = self.forward(y)
        diff = (emb2 - emb1)**2
        diff_norm = torch.norm(diff)
        
        if diff_norm.item() > threshold:
            return(0)
        
        elif diff_norm.item() <= threshold:
            return(1)
        
        return(-1)