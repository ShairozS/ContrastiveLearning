import torch
import numpy as np

class Trainer:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 optimizer = None,
                 lr = 0.005,
                 loss_functions=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        self.model = model.to(device)
        self.dataloader = dataloader
        
        ## Initialize the optimizer if none has been passed in
        if optimizer is None:
            self.optimizer = torch.nn.optimize


        ## Initialize the loss function(s)


    ##########################
    #                        #
    #  Single Iter Training  #
    #                        #
    ##########################


    def train_iter(self, input, label, verbose=0):
        pass

        ## Zero the gradients


        ## Pass the inputs through the model


        ## Calculate the loss(es)


        ## Verbose printouts


        ## Pass the loss backward


        ## Take an optimizer step


        ## Return the total loss


    ##########################
    #                        #
    #  Mutlti Epoch Training #
    #                        #
    ##########################



    def train(self,
              epochs,
              print_every=1,
              error_tolerance=5,
              writer=None):

        pass
        ## Loop over epochs in the range of epochs



            ## If the report_every epoch is reached, reinitialize metric lists



            ## Enumerate self.dataloader


                ## Grab an example


                ## Send it to self.device


                ## Try to train_iter


                ## If not, hit the exception message, increment error counter


                ## Update the metric lists and counters


            ## If we've hit report_every epoch, print the report



            ## Write the outputs to Tensorboard if writer is not None
