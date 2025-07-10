import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, labels, targeted=True):
        super(CELoss, self).__init__()
        self.labels=labels
        self.targeted=targeted
        self.ce=nn.CrossEntropyLoss(reduction='mean')
        self.labels.requires_grad = False
    def forward(self, logits):
        loss = self.ce(logits, self.labels)
        if self.targeted==False:
            loss = -loss
        return -loss
    

class LogitLoss(torch.nn.Module):
    def __init__(self, labels, targeted=True):
        super(LogitLoss, self).__init__()
        self.labels=labels
        self.targeted=targeted
        self.labels.requires_grad = False
    def forward(self, logits):
        logit_dists = logits.gather(1,self.labels.view(-1, 1))
        loss = logit_dists.sum()
        if self.targeted==False:
            loss=-loss
        return loss
    

class TopkLoss(torch.nn.Module):
    def __init__(self, labels, targeted=True, top_k=None, p=2):
        super(TopkLoss, self).__init__()
        self.labels=labels
        self.targeted=targeted
        self.labels.requires_grad=False
        self.top_k = top_k
        self.p = p
        self.ce=nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits):
        topk_logits, _ = torch.topk(logits, k=self.top_k, dim=1)
        nontg_logits = torch.mean(topk_logits[:, 1:], dim=1)
        topk_total = logits[:, self.labels[0]] - nontg_logits

        loss = topk_total.mean()

        if not self.targeted:
            loss = -loss  
        return loss


class Topkdis(torch.nn.Module):
    def __init__(self, labels, targeted=True, top_k=None, p=2):
        super(Topkdis, self).__init__()
        self.labels=labels
        self.targeted=targeted
        self.labels.requires_grad=False
        self.top_k = top_k
        self.p = p

    def forward(self, inputs, logits):
        topk_logits, _ = torch.topk(logits, k=self.top_k, dim=1)
        nontg_logits = torch.mean(topk_logits[:, 1:], dim=1)
        tg_logits = logits[:, self.labels[0]]
        nontg_grads = torch.autograd.grad(nontg_logits.mean(), inputs, retain_graph=True, create_graph=False)[0]
        tg_grads = torch.autograd.grad(tg_logits.mean(), inputs, retain_graph=True, create_graph=False)[0]
        f = tg_logits - nontg_logits 
        grad_f = tg_grads - nontg_grads
        dis = f / torch.norm(grad_f, p=2, dim=[1,2,3], keepdim=False)
        loss = dis.mean()
        if not self.targeted:
            loss = -loss 
        return loss
     