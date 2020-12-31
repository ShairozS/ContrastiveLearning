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
                 lr = 0.0005,
                 loss_function=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        self.model = model.to(device)
        self.dataloader = dataloader
        
        
        ## Initialize the optimizer if none has been passed in
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        else:
            self.optimzer = optimizer


        ## Initialize the loss function(s)
        self.loss_function = loss_function
        
        self.device = device
        
        self.curr_epoch = 0

    ##########################
    #                        #
    #  Single Iter Training  #
    #                        #
    ##########################


    def train_iter(self, input, label, verbose=0):

        ## Zero the gradients
        self.optimizer.zero_grad()

        ## Pass the inputs through the model
        x1, x2 = input
        emb1 = self.model(x1); emb2 = self.model(x2)
        
        ## Calculate the loss(es)
        loss = self.loss_function(emb1, emb2, label)

        ## Verbose printouts

        ## Pass the loss backward
        loss.backward()

        ## Take an optimizer step
        self.optimizer.step()

        ## Return the total loss
        return(loss)

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

        
        ## Loop over epochs in the range of epochs
        for epoch in range(self.curr_epoch, self.curr_epoch + epochs):
            
            epoch_losses = []

            ## If the report_every epoch is reached, reinitialize metric lists
            if epoch % print_every == 0:
                print("----- Epoch: " + str(epoch) + " -----")
    

            ## Enumerate self.dataloader
            for idx, data_dict in enumerate(self.dataloader):

                ## Grab an example
                x1 = data_dict["x1"]; x2 = data_dict["x2"]
                y = data_dict["label"]

                ## Send it to self.device
                x1 = x1.to(self.device); x2 = x2.to(self.device)
                y = y.to(self.device)
                
                ## Try to train_iter
                batch_loss = self.train_iter((x1,x2), y)

                ## If not, hit the exception message, increment error counter


                ## Update the metric lists and counters
                epoch_losses.append(batch_loss.item())
                
                
            self.curr_epoch += 1
            
    
            ## If we've hit report_every epoch, print the report
            if epoch % print_every == 0:
                print("loss: " + str(np.mean(epoch_losses)))