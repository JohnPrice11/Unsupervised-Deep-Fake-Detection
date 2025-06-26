import torch
import torch.nn as nn
import torch.nn.functional as F  


class ECLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ECLoss, self).__init__()
         
        self.temperature = max(temperature, 1e-8)
        self.contrast_mode = contrast_mode
        self.base_temperature = max(base_temperature, 1e-8)  

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: shape [bsz, n_views, feature_dim] or [bsz, n_views, H, W]
                      (will be flattened to [bsz, n_views, feature_dim])
            labels: ground truth labels for each sample in the batch, shape [bsz]
            mask: contrastive mask, shape [bsz, bsz]
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

         
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
         
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
             
             
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
             
             
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
             
            mask = mask.float().to(device)

         
         
        contrast_count = features.shape[1]  
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

         
        if self.contrast_mode == 'one':
             
            anchor_feature = features[:, 0]  
            anchor_count = 1  
        elif self.contrast_mode == 'all':
             
            anchor_feature = contrast_feature  
            anchor_count = contrast_count  
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
         
        anchor_feature = F.normalize(anchor_feature, dim=1)
        contrast_feature = F.normalize(contrast_feature, dim=1) 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  

         
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

         
         
         
        mask = mask.repeat(anchor_count, contrast_count)

         
         
        logits_mask = torch.scatter(
            torch.ones_like(mask),  
            1,  
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  
            0  
        )

         
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits_sum = exp_logits.sum(1, keepdim=True)

         
        log_prob = logits - torch.log(exp_logits_sum + 1e-9)


        positive_log_probs = (mask * log_prob)  
        sum_positive_log_probs = positive_log_probs.sum(1)
        
         
         
        num_positives = mask.sum(1)

        mean_log_prob_pos = sum_positive_log_probs / num_positives.clamp(min=1e-9)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

         

        loss = loss.mean()


        return loss