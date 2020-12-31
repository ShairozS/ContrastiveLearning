from torch import nn
import torch

class ContrastiveLoss(torch.nn.Module):
    
    
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    args:
        distance (function): A function that returns the distance between two tensors - should be a valid metric over R; default=nn.PairwiseDistance()
        margin (scalar): The margin value between positive and negative class ; default=1.0
        pos_class_weight (scalar): The weight to put on positive class loss when positive classes are sparse ; default=20
    """

    
    def __init__(self,
                 distance = nn.PairwiseDistance(),
                 margin=1.0,
                 pos_class_weight=20,
                 reduction='mean'):
        
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.pos_class_weight = pos_class_weight
        self.reduction = reduction

        
        
    def forward(self, x, y, label):
        '''
        Return the contrastive loss between two similar or dissimilar outputs
        
        Args:
            x (torch.Tensor) : The first input tensor (B, N)
            y (torch.Tensor) : The second input tensor (B,N)
            label (torch.Tensor) : A tensor with elements either 0 or 1 indicating dissimilar or similar (B, 1)
        '''
        
        assert x.shape==y.shape, str(x.shape) + "does not match input 2: " + str(y.shape)
        distance = self.distance(x, y)
        
        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings and a margin
        if self.reduction == 'mean':
            loss_contrastive = torch.mean((label) * torch.pow(distance, 2) +
                                      (self.pos_class_weight*(1-label)) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        if self.reduction == 'sum':
            loss_contrastive = torch.sum((label) * torch.pow(distance, 2) +
                                      (self.pos_class_weight*(1-label)) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))


        return loss_contrastive

    
class TripletLoss(torch.nn.Module):
    
    '''
    Triplet loss function.
    Based on: 
    
    Args:
        distance (function): A function that returns the distance between two tensors - should be a valid metric over R; default=nn.PairwiseDistance()
        margin (scalar): The margin value between positive and negative class ; default=1.0
        pos_class_weight (scalar): The weight to put on positive class loss when positive classes are sparse ; default=20 
        
    '''
    
    def __init__(self, 
                 distance = nn.PairwiseDistance(),
                 margin = 1.0,
                 pos_class_weight = 1,
                 reduction = 'mean'):
        
        super(TripletLoss, self).__init__()
        self.distance = distance
        self.margin = margin
        self.pos_class_weight = pos_class_weight
        self.reduction = reduction
        
    
    def forward(self, anchor, positive, negative):
        '''
        Return the triplet loss between 3 data elements
        
        Args:
            anchor (torch.Tensor) : The anchor example as a batched tensor (B, N)
            positive (torch.Tensor) : The similar examples to the anchor as a batched tensor (B, N)
            negative (torch.Tensor) : The dissimilar examples to the anchor as a batched tensor (B, N)
            
        Returns:
            (torch.Tensor): The calculated batched triplet loss
        
        '''
        ap_distance = self.distance(anchor, positive)
        an_distance = self.distance(anchor, negative)
        distance = ap_distance - an_distance + self.margin
        
        return(torch.max(distance, 0))