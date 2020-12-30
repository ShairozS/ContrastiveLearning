from torch import nn


class TemplateLoss(nn.Module):


      def __init__(self, hyper_params):

            '''
            Initialize the loss function here with any specific hyperparameters
            '''
            pass



      def forward(self, y, yhat):

            '''
            Return the scalar loss given two inputs y and yhat
            '''

            ## Some processing here
            loss = torch.sqrt(torch.sum((y - yhat)**2))


            return(loss)
