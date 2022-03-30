import torch
import torch.nn as nn
import torch.nn.functional as F

class GLOBE_SupLoss(nn.Module):
    def __init__(self, base_temp=0.07, temp=0.07):
        super().__init__()
        self.base_temp = base_temp
        self.temp = temp

    def forward(self, features, labels):
        pass
        # after publishment
        
class GLOBE_UnsupLoss(nn.Module):
    def __init__(self, base_temp=0.07, temp=0.07):
        super().__init__()
        self.base_temp = base_temp
        self.temp = temp

    def forward(self, features, mask):
        pass 
        # after publishment

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