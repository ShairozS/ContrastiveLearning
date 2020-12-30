from torch import nn


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

        
        
    def forward(self, output1, output2, label):
        
        euclidean_distance = self.distance(output1, output2)
        
        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings and a margin
        if self.reduction == 'mean':
            loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (self.pos_class_weight*(1-label)) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if self.reduction == 'sum':
            loss_contrastive = torch.sum((label) * torch.pow(euclidean_distance, 2) +
                                      (self.pos_class_weight*(1-label)) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
