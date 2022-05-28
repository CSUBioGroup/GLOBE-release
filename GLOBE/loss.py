import torch
import torch.nn as nn
import torch.nn.functional as F


# A very small modification from SupconLoss[1]
# [1]: Khosla, Prannay, et al. "Supervised contrastive learning." arXiv preprint arXiv:2004.11362 (2020).
class GLOBE_SupLoss(nn.Module):
    def __init__(self, base_temp=0.07, temp=0.07):
        super().__init__()
        self.base_temp = base_temp
        self.temp = temp

    def forward(self, features, labels):
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to('cuda')

        # ensure the features are L2-normalized
        sample_sim = torch.div(
                torch.matmul(features, features.T),
                self.temp
        )

        # if torch.any(torch.isnan(sample_sim)):
        #     torch.save('/home/yxh/f.pt')
        #     res = torch.matmul(features, features.T)
        #     torch.save(res, '/home/yxh/ff.pt')
        #     raise ValueError('Nan')

        # print('sample_sim ,', torch.any(torch.isnan(sample_sim)))

        # for numerical stability
        logits_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
        logits = sample_sim - logits_max.detach()

        # tile mask, mask used for filling the diagonal with 0
        logits_mask = torch.scatter(
                torch.ones_like(mask),
                1, 
                torch.arange(batch_size).view(-1, 1).to('cuda'),
                0
        )
        mask = mask * logits_mask  

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask   # exclude diagonal
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # add 1e-20 to avoid inexistence of positive anchors
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)  

        # print('mean_log_prob_pos ,', torch.any(torch.isnan(mean_log_prob_pos)))

        # loss
        loss = - (self.temp / self.base_temp) * mean_log_prob_pos
        loss = loss.mean()

        return loss

# Same implementation as paper, and another approach is also incorporated
class GLOBE_UnsupLoss(nn.Module):
    def __init__(self, base_temp=0.07, temp=0.07):
        super().__init__()
        self.base_temp = base_temp
        self.temp = temp

    def forward(self, features, mask):
        '''
            features: (n_bsz, 2*n_features, 1), n_bsz_anchor_features + n_bsz_positive_features
            mask: (2*bsz, 2*bsz) or (bsz, bsz)
        '''

        batch_size = features.shape[0]

        if not (mask.shape[0] == batch_size*2 or mask.shape[0] == batch_size) :
            raise ValueError('Num of labels does not match num of features')
        # mask = torch.eq(labels, labels.T).float().to('cuda')

        # flatten the features
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # n_anchor concat n_positive

        # ensure the features are L2-normalized
        sample_sim = torch.div(
                torch.matmul(features, features.T),
                self.temp
        )


        # for numerical stability
        logits_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
        logits = sample_sim - logits_max.detach()

        # tile mask, mask used for filling the diagonal with 0
        repeat_times = batch_size * 2 // mask.shape[0] 
        mask = mask.repeat(repeat_times, repeat_times)  
        logits_mask = torch.scatter(
                torch.ones_like(mask),
                1, 
                torch.arange(batch_size).view(-1, 1).to('cuda'),
                0
        )
        mask = mask * logits_mask  

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask   # exclude diagonal
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # add 1e-20 to avoid inexistence of positive anchors
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  

        # loss
        loss = - (self.temp / self.base_temp) * mean_log_prob_pos
        loss = loss.mean()

        return loss

# InfoNCE
# copied from SMILE, changed the input
class InfoNCE(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.temperature = temperature
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, 
                                                           dtype=bool)).float())
            
    def forward(self, features):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        #z_i = F.normalize(emb_i, dim=1)
        #z_j = F.normalize(emb_j, dim=1)
        
        # z_i = F.normalize(emb_i, dim=1,p=2)
        # z_j = F.normalize(emb_j, dim=1,p=2)

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        representations = F.normalize(features, dim=1,p=2)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)

        # negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).float().cuda()
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss