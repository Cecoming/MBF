import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, ignore_index=None, reduction='mean', class_weights=None):
        super(KLabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reduction = reduction
        if class_weights is not None:
            self.class_weights = class_weights.view(1, class_weights.shape[0], 1, 1)
        else:
            self.class_weights = None

    def forward(self, input, target):
        # The shape of input is [B, K+1, H, W], where K is the number of classes excluding the background class
        # The shape of target is [B, H, W], with values ranging from 0 to K, where 0 represents the background class
        B, C, H, W = input.size()
        
        # Apply softmax operation to obtain probability distribution
        log_probs = F.log_softmax(input, dim=1)
        
        # Generate a tensor with smoothed label values
        smoothed_labels = torch.full((B, C, H, W), self.epsilon / (C - 1)).to(input.device)
        
        # Set the positions of target labels to 1 - epsilon
        target = target.unsqueeze(1)  # [B, 1, H, W] Add an extra dimension to match the dimension of input
        smoothed_labels.scatter_(1, target, 1 - self.epsilon)
        
        # Apply ignore_index to ignore the background class
        # mask = target != self.ignore_index  # Mask to ignore background values
        if self.ignore_index is not None:
            mask = target != self.ignore_index  # Mask to ignore background values
            smoothed_labels = smoothed_labels * mask.float()

        # If class weights are provided, use them to adjust the loss
        if self.class_weights is not None:
            class_weights = self.class_weights.to(input.device)
            smoothed_labels = smoothed_labels * class_weights
        
        # Calculate the label smoothed cross entropy loss
        loss = -log_probs * smoothed_labels
        
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            scale = smoothed_labels.shape[0] * smoothed_labels.shape[2] * smoothed_labels.shape[3]
            loss = loss.sum() / scale
        elif self.reduction == 'none':
            loss = loss.sum(dim=1)

        return loss

if __name__ == '__main__':
    # Example usage
    criteria = KLabelSmoothedCrossEntropyLoss(epsilon=0.1, ignore_index=0)
    logits = torch.randn(2, 3, 4, 5)  # Simulated input
    target = torch.ones(2, 4, 5, dtype=torch.long)  # Simulated target
    loss = criteria(logits, target)
    print(loss)
