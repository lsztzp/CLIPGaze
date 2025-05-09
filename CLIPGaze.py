import torch
import torch.nn.functional as F
from torch import nn, Tensor
from positional_encodings import PositionEmbeddingSine2d


class CLIPGaze(nn.Module):
    def __init__(self, transformer, spatial_dim, dropout=0.2, max_len = 7, project_num  = 16, device = "cuda:0"):
        super(CLIPGaze, self).__init__()
        self.spatial_dim = spatial_dim
        self.transformer = transformer.to(device)
        self.hidden_dim = transformer.d_model
        #fixation embeddings
        self.querypos_embed = nn.Embedding(max_len,self.hidden_dim).to(device)
        #2D patch positional encoding
        self.patchpos_embed = PositionEmbeddingSine2d(spatial_dim, hidden_dim=self.hidden_dim, normalize=True, device = device)
        #2D pixel positional encoding for initial fixation
        self.queryfix_embed = PositionEmbeddingSine2d((spatial_dim[0] * project_num, spatial_dim[1] * project_num), hidden_dim=self.hidden_dim, normalize=True, flatten = False, device = device).pos.to(device)
        #classify fixation, or PAD tokens
        self.token_predictor = nn.Linear(self.hidden_dim, 2)
        #Gaussian parameters for x,y,t
        self.generator_y_mu = nn.Linear(self.hidden_dim, 1).to(device)
        self.generator_x_mu = nn.Linear(self.hidden_dim, 1).to(device)

        self.generator_t_mu = nn.Linear(self.hidden_dim, 1).to(device)

        self.device = device
        self.max_len = max_len
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.LogSoftmax(dim=-1).to(device)

        #projection for first fixation encoding
        self.firstfix_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, src: Tensor, tgt: Tensor, task: Tensor):
        cond = task
        activations = src
        a, b = len(activations), len(activations[0])
        activations = [torch.stack([activations[i][j].squeeze(0) for i in range(a)]).permute(1,0,2) for j in range(b)]
        tgt_input = torch.zeros(self.max_len, cond.size(0), self.hidden_dim).to(self.device)#Notice that this where we convert target input to zeros

        tgt_input[0, :, :] = self.firstfix_linear(self.queryfix_embed[tgt[:, 0], tgt[:,1], :])
        outs = self.transformer(src=activations, tgt=tgt_input, tgt_mask= None, tgt_key_padding_mask = None,
        task = cond.to(self.device), querypos_embed = self.querypos_embed.weight.unsqueeze(1), patchpos_embed = self.patchpos_embed)

        outs = self.dropout(outs)
        y_mu, x_mu, t_mu = self.generator_y_mu(outs), self.generator_x_mu(outs), self.generator_t_mu(outs)

        return self.softmax(self.token_predictor(outs)), self.activation(y_mu), self.activation(x_mu), self.activation(t_mu)



