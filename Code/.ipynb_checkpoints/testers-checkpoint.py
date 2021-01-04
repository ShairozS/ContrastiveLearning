import torch
import numpy as np

class Tester:

    ##########################
    #                        #
    #     Initialization     #
    #                        #
    ##########################



    def __init__(self,
                 model,
                 dataloader,
                 metric=None,
                 device='cuda'):
        

        
        ## Assign class attributes and send model to device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.metric = metric
        self.device = device
        self.curr_epoch = 0

        
    ##########################
    #                        #
    #  Single Iter Testing   #
    #                        #
    ##########################


    def test_batch(self, input, label):

        ## Pass the inputs through the model
        x1, x2 = input
        res = model.predict(x1,x2)
        
        ## Calculate the metric
        metric = self.metric(res, label)

        ## Return the metric
        return(metric)

    ##########################
    #                        #
    #  Mutlti Epoch Testing  #
    #                        #
    ##########################



    def test(self):
        
        batch_metrics = []

        ## Enumerate self.dataloader
        for idx, data_dict in enumerate(self.dataloader):

            ## Grab an example
            x1 = data_dict["x1"]; x2 = data_dict["x2"]
            y = data_dict["label"]

            ## Send it to self.device
            x1 = x1.to(self.device); x2 = x2.to(self.device)
            y = y.to(self.device)

            ## Try to train_iter
            batch_metric = self.test_batch((x1,x2), y)

            ## Update the metric lists and counters
            batch_metrics.append(batch_metric.item())
            
        return(np.mean(batch_metrics))
