import time
import torch
import numpy as np
#from torchmetrics.regression import ConcordanceCorrCoef
    
def accuracy_av(output, target):
    with torch.no_grad():
        # Assuming output and target are both torch tensors
        batch_size = target.size(0)

        # Calculate the Concordance Correlation Coefficient (CCC)
        cc = concordance_corr_coef(target, output)

        # Calculate the average of the Arousal and Valence CCCs
        avg_cc = cc.mean() / batch_size

        # Convert the average CCC to a scalar and put it into a list
        return avg_cc

def concordance_corr_coef(y_true, y_pred):
    # print(len(y_true))
    # print(len(y_pred))
    with torch.no_grad():

        y_true = y_true.float()  # Ensures y_true is floating-point
        y_pred = y_pred.float()  # Ensures y_pred is floating-point

        # Mean for each variable
        mean_true = torch.mean(y_true, dim=0)
        mean_pred = torch.mean(y_pred, dim=0)

        # Variance for each variable
        var_true = torch.var(y_true, dim=0, unbiased=False)
        var_pred = torch.var(y_pred, dim=0, unbiased=False)

        # Covariance between y_true and y_pred
        covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=0)

        # Pearson correlation coefficient
        pearson_corr = covariance / (torch.sqrt(var_true) * torch.sqrt(var_pred))

        # Concordance Correlation Coefficient
        ccc = (2 * pearson_corr * torch.sqrt(var_true) * torch.sqrt(var_pred)) / \
            (var_true + var_pred + (mean_true - mean_pred)**2)

        return ccc

# def accuracy_au(output, target, topk=(1,)): 

     
#     with torch.no_grad():
        
#         # calculate F1 here 
#         #run this over a loop of the batch size 
#         f1 = multiclass_f1_score(output, target, num_classes=12, average="macro")

#         return f1


def accuracy_au(output, target, num_classes=12):
    with torch.no_grad():

        #USING F1 FOR AU

        # Initialize tensors to hold true positives, false positives, false negatives
        true_positives = torch.zeros(num_classes)
        false_positives = torch.zeros(num_classes)
        false_negatives = torch.zeros(num_classes)

        # Calculate TP, FP, FN for each class
        for i in range(num_classes):
            true_positives[i] = torch.sum((output == i) & (target == i))
            false_positives[i] = torch.sum((output == i) & (target != i))
            false_negatives[i] = torch.sum((output != i) & (target == i))

        # Calculate precision, recall, and F1 for each class
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

        # average F1 score
        avgf1 = torch.mean(f1_scores)

        return avgf1


        
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        """Computes the precision@k for the specified values of k"""
        # print("Output: ",output)
        # print("target: ",target)
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum().item()
                res.append(correct_k*100.0 / batch_size)

            if len(res)==1:
                return res[0]
            else:
                return res



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count


class Timer(object):
    """
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval
    


    