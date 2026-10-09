import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
try:
    import dgl
    from dgl.nn import GATConv
except ImportError:
    dgl = None
    GATConv = None
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
from src.sctm import *
from src.tcn import *
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
#from mamba_ssm import Mamba

torch.manual_seed(1)


## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
    def __init__(self, feats):
        super(LSTM_Univariate, self).__init__()
        self.name = 'LSTM_Univariate'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 1
        self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

    def forward(self, x):
        hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64),
                   torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
        outputs = []
        for i, g in enumerate(x):
            multivariate_output = []
            for j in range(self.n_feats):
                univariate_input = g.view(-1)[j].view(1, 1, -1)
                out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
                multivariate_output.append(2 * out.view(-1))
            output = torch.cat(multivariate_output)
            outputs.append(output)
        return torch.stack(outputs)


## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
    def __init__(self, feats):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5  # MHA w_size = 5
        self.n = self.n_feats * self.n_window
        self.atts = [nn.Sequential(nn.Linear(self.n, feats * feats),
                                   nn.ReLU(True)) for i in range(1)]
        self.atts = nn.ModuleList(self.atts)

    def forward(self, g):
        for at in self.atts:
            ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
            g = torch.matmul(g, ats)
        return g, ats


## LSTM_AD Model
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super(LSTM_AD, self).__init__()
        self.name = 'LSTM_AD'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden)
        self.lstm2 = nn.LSTM(feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (
            torch.rand(1, 1, self.n_hidden, dtype=torch.float64, device=x.device),
            torch.randn(1, 1, self.n_hidden, dtype=torch.float64, device=x.device))
        hidden2 = (
            torch.rand(1, 1, self.n_feats, dtype=torch.float64, device=x.device),
            torch.randn(1, 1, self.n_feats, dtype=torch.float64, device=x.device))
        outputs = []
        for i, g in enumerate(x):
            out, hidden = self.lstm(g.view(1, 1, -1), hidden)
            out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
            out = self.fcn(out.view(-1))
            outputs.append(2 * out.view(-1))
        return torch.stack(outputs)


## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 5  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden


## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5  # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1, -1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        self.g = dgl.graph((torch.tensor(list(range(1, feats + 1))), torch.tensor([0] * feats)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(feats, 1, feats)
        self.time_gat = GATConv(feats, 1, feats)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1)

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats), data))
        feat_r = self.feature_gat(self.g, data_r)
        data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
        time_r = self.time_gat(self.g, data_t)
        data = torch.cat((torch.zeros(1, self.n_feats), data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        x, h = self.gru(x, hidden)
        return x.view(-1), h


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5  # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)


# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
    def __init__(self, feats):
        super(TranAD_Basic, self).__init__()
        self.name = 'TranAD_Basic'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x


# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
    def __init__(self, feats):
        super(TranAD_Transformer, self).__init__()
        self.name = 'TranAD_Transformer'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_hidden = 8
        self.n_window = 10
        self.n = 2 * self.n_feats * self.n_window
        self.transformer_encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        tgt = self.transformer_encoder(src)
        return tgt

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c, tgt))
        x1 = x1.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x1 = self.fcn(x1)
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c, tgt))
        x2 = x2.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x2 = self.fcn(x2)
        return x1, x2


# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
    def __init__(self, feats):
        super(TranAD_Adversarial, self).__init__()
        self.name = 'TranAD_Adversarial'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode_decode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x = self.encode_decode(src, c, tgt)
        # Phase 2 - With anomaly scores
        c = (x - src) ** 2
        x = self.encode_decode(src, c, tgt)
        return x


# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
    def __init__(self, feats):
        super(TranAD_SelfConditioning, self).__init__()
        self.name = 'TranAD_SelfConditioning'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class EPAADNet(nn.Module):
    def __init__(self, feats):
        super(EPAADNet, self).__init__()
        self.name = 'EPAADNet'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(num_outputs=feats, kernel_size=4, dropout=0.2)  # K=3&4 (Batch, output_channel, seq_len)
        self.n = self.n_feats * self.n_window
        self.sctm = SCTM(feats, feats)
        self.dropout1 = nn.Dropout(0.1)

        # 自定义 decoder 层 (含 2 个 autoencoder + cross-attn + FFN), 直接调用, 不用 TransformerDecoder 包装
        self.decoder_layer = TransformerDecoderLayer1(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)

        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt, training=True):
        # src: (T, B, C) -> TCN 需要 (B, C, T)
        src2 = self.l_tcn(src.permute(1, 2, 0))
        src = src + self.dropout1(src2.permute(2, 0, 1))
        src = self.sctm(src)

        x = self.decoder_layer(tgt, src)
        x = self.fcn(x)

        return x


class CNNLSTM(nn.Module):
    def __init__(self, feats, hidden_size=64, num_layers=2, dropout=0.2):
        super(CNNLSTM, self).__init__()

        self.name = 'CNNLSTM'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10

        # CNN
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_window, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # LSTM
        self.lstm = nn.LSTM(feats, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, feats)
        self.BN1 = nn.BatchNorm1d(self.n_window)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, feats)
        self.BN2 = nn.BatchNorm1d(feats)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(1, 0, 2)
        N = x.size(0)
        # print('N', N)
        # print('输入LSTM', x.shape)
        x_lstm = self.lstm(x)[0]
        # print('lstm处理后', x_lstm.shape)
        x_lstm = self.relu1(self.BN1(self.fc1(x_lstm)))
        # print('调整大小之后', x_lstm.shape)
        x_lstm = x_lstm.mean(1)
        # print('第一个维度平均', x_lstm.shape)
        x_lstm = x_lstm.unsqueeze(1)
        # print('增加一个维度', x_lstm.shape)
        # print('输出LSTM', x_lstm.shape)

        # print('*****************************')
        # print('输入CNN', x.shape)
        x_cnn = self.conv_block1(x)
        # print('经过第一个卷积块', x_cnn.shape)
        x_cnn = self.conv_block2(x_cnn)
        # print('经过第二个卷积块', x_cnn.shape)
        x_cnn = self.conv_block3(x_cnn)
        # print('经过第三个卷积块', x_cnn.shape)
        x_cnn = torch.mean(x_cnn, 2)
        # print('第二个维度平均', x_cnn.shape)
        x_cnn = self.relu2(self.BN2(self.fc2(x_cnn)))
        # print('调整大小', x_cnn.shape)
        x_cnn = x_cnn.unsqueeze(1)
        # print('增加一个维度', x_cnn.shape)
        # print('输出CNN', x_cnn.shape)

        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = torch.mean(x, 1)
        x = x.unsqueeze(1)
        # print('CNNLSTM拼接后平均：', x.shape)

        x = x.permute(1, 0, 2)
        # print('调整', x.shape)

        return x


class CNNLSTM_RH(nn.Module):
    def __init__(self, feats, hidden_size=64, num_layers=2, dropout=0.2):
        super(CNNLSTM_RH, self).__init__()

        self.name = 'CNNLSTM_RH'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window1 = 10
        self.n_window2 = 20
        self.n_window3 = 30

        # CNN
        self.conv_block11 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_window1, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        self.conv_block12 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_window2, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )
        self.conv_block13 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_window3, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # LSTM
        self.lstm = nn.LSTM(feats, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, feats)
        self.BN1 = nn.BatchNorm1d(self.n_window1)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, feats)
        self.BN2 = nn.BatchNorm1d(feats)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x1, x2, x3):
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        x3 = x3.permute(1, 0, 2)
        print('x1,x2,x3', x1.shape, x2.shape, x3.shape)

        N = x1.size(0)
        # print('N', N)
        # print('输入LSTM', x.shape)
        x_lstm = self.lstm(x1)[0]
        # print('lstm处理后', x_lstm.shape)
        x_lstm = self.relu1(self.BN1(self.fc1(x_lstm)))
        # print('调整大小之后', x_lstm.shape)
        x_lstm = x_lstm.mean(1)
        # print('第一个维度平均', x_lstm.shape)
        x_lstm = x_lstm.unsqueeze(1)
        # print('增加一个维度', x_lstm.shape)
        print('输出LSTM', x_lstm.shape)

        print('*************111****************')
        # print('输入CNN', x.shape)
        x_cnn1 = self.conv_block11(x1)
        # print('经过第一个卷积块', x_cnn.shape)
        x_cnn1 = self.conv_block2(x_cnn1)
        # print('经过第二个卷积块', x_cnn.shape)
        x_cnn1 = self.conv_block3(x_cnn1)
        # print('经过第三个卷积块', x_cnn.shape)
        x_cnn1 = torch.mean(x_cnn1, 2)
        # print('第二个维度平均', x_cnn.shape)
        x_cnn1 = self.relu2(self.BN2(self.fc2(x_cnn1)))
        # print('调整大小', x_cnn.shape)
        x_cnn1 = x_cnn1.unsqueeze(1)
        # print('增加一个维度', x_cnn.shape)
        print('输出CNN1', x_cnn1.shape)

        print('*************222****************')
        # print('输入CNN', x.shape)
        x_cnn2 = self.conv_block12(x2)
        # print('经过第一个卷积块', x_cnn.shape)
        x_cnn2 = self.conv_block2(x_cnn2)
        # print('经过第二个卷积块', x_cnn.shape)
        x_cnn2 = self.conv_block3(x_cnn2)
        # print('经过第三个卷积块', x_cnn.shape)
        x_cnn2 = torch.mean(x_cnn2, 2)
        # print('第二个维度平均', x_cnn.shape)
        x_cnn2 = self.relu2(self.BN2(self.fc2(x_cnn2)))
        # print('调整大小', x_cnn.shape)
        x_cnn2 = x_cnn2.unsqueeze(1)
        # print('增加一个维度', x_cnn.shape)
        print('输出CNN2', x_cnn2.shape)

        x = torch.cat([x_cnn1, x_lstm], dim=1)
        x = torch.mean(x, 1)
        x = x.unsqueeze(1)
        # print('CNNLSTM拼接后平均：', x.shape)

        x = x.permute(1, 0, 2)
        # print('调整', x.shape)

        return x


# class CustomMamba(Mamba):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def forward(self, hidden_states, inference_params=None):
#         # 将 self.in_proj.weight 转换为 torch.nn.Parameter 对象
#         self.in_proj.weight = torch.nn.Parameter(self.in_proj.weight.float())
#         return super().forward(hidden_states, inference_params)


# 使用 CustomMamba 类代替 Mamba 类
# self.mamba1 = CustomMamba(d_model=38, d_state=256, d_conv=2, expand=1)

# ==================== TimesNet (ICLR 2023) ====================
# Core idea: FFT finds dominant periods → reshape 1D→2D → Inception 2D conv
class InceptionBlockV1(nn.Module):
    """2D Inception block for TimesNet: multi-scale 2D convolution."""
    def __init__(self, in_channels, out_channels):
        super(InceptionBlockV1, self).__init__()
        # Multi-scale 2D conv kernels
        mid = max(out_channels // 4, 1)
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, mid, kernel_size=5, padding=2)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels - 3 * mid, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out_pool = self.maxpool_conv(x)
        out = torch.cat([out1, out3, out5, out_pool], dim=1)
        return self.relu(self.bn(out))


class TimesBlock(nn.Module):
    """
    TimesNet core block:
    1. FFT to find top-k periods from the frequency domain
    2. Reshape 1D sequence → 2D map (period × frequency) per period
    3. Inception 2D conv to capture intra- & inter-period patterns
    4. Adaptive weighted fusion across periods
    """
    def __init__(self, seq_len, d_model, top_k=3):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.top_k = min(top_k, seq_len // 2)
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_model // 2),
            nn.GELU(),
            InceptionBlockV1(d_model // 2, d_model),
        )
        # Learnable fusion weights (instead of softmax on FFT amplitudes)
        self.fusion_weight = nn.Parameter(torch.ones(self.top_k) / self.top_k)

    def forward(self, x):
        # x: (B, C, T)  where C=d_model, T=seq_len
        B, C, T = x.shape

        # 1. FFT to find dominant periods
        x_fft = torch.fft.rfft(x, dim=-1)          # (B, C, T//2+1)
        amplitudes = torch.abs(x_fft).mean(dim=1)   # (B, T//2+1)  avg over channels
        # Exclude DC component (index 0), select top_k frequencies
        amps = amplitudes[:, 1:]                     # (B, T//2)
        if amps.shape[1] < self.top_k:
            top_k_eff = amps.shape[1]
        else:
            top_k_eff = self.top_k
        _, top_indices = torch.topk(amps, top_k_eff, dim=-1)  # (B, top_k_eff), 0-indexed
        # Convert frequency index → period: period = T / (freq_idx + 1)
        periods = T / (top_indices.float() + 1.0)   # (B, top_k_eff)

        # 2. For each period (use batch-mean for consistency)
        avg_periods = periods.mean(dim=0)            # (top_k_eff,)
        outs = []
        for i in range(top_k_eff):
            p_raw = avg_periods[i].item()
            p = max(2, min(T, int(round(p_raw))))    # clamp to valid range
            # Make sure T is divisible: pad if needed
            if T % p == 0:
                pad = 0
                x_pad = x
            else:
                pad = p - (T % p)
                x_pad = F.pad(x, (0, pad))           # (B, C, T+pad)
            T_pad = T + pad
            # Reshape to 2D: (B, C, p, T_pad//p)
            x_2d = x_pad.reshape(B, C, p, T_pad // p)
            # 3. 2D Inception conv
            x_2d = self.conv(x_2d)                   # (B, C, p, T_pad//p)
            # Reshape back to 1D
            x_1d = x_2d.reshape(B, C, T_pad)         # (B, C, T_pad)
            if pad > 0:
                x_1d = x_1d[:, :, :T]               # (B, C, T)
            outs.append(x_1d)

        # 4. Weighted fusion (learnable weights)
        outs = torch.stack(outs, dim=-1)             # (B, C, T, top_k_eff)
        w = F.softmax(self.fusion_weight[:top_k_eff], dim=0)  # (top_k_eff,)
        w = w.view(1, 1, 1, -1)
        x = torch.sum(outs * w, dim=-1)              # (B, C, T)
        return x


class TimesNet(nn.Module):
    """TimesNet: Temporal 2D-Variation Modeling for Time Series Anomaly Detection.
    Adapted from https://arxiv.org/abs/2210.02186 (ICLR 2023).
    Uses FFT-based period discovery + 2D Inception convolution for reconstruction.
    """
    def __init__(self, feats):
        super(TimesNet, self).__init__()
        self.name = 'TimesNet'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.d_model = 32
        self.top_k = 3
        self.num_blocks = 2

        self.embed = nn.Linear(feats, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1, self.n_window)
        self.blocks = nn.ModuleList([
            TimesBlock(self.n_window, self.d_model, self.top_k)
            for _ in range(self.num_blocks)
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, feats)
        self.fcn = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, C) where T=n_window
        B, T, C = x.shape
        # Embed features
        x = self.embed(x)                           # (B, T, d_model)
        # Positional encoding expects (T, B, d_model)
        x = x.permute(1, 0, 2)                      # (T, B, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 2, 0)                      # (B, d_model, T)
        # TimesBlocks (operate on (B, d_model, T))
        for block in self.blocks:
            x = x + block(x)                         # residual
        # Back to (B, T, d_model)
        x = x.permute(0, 2, 1)                      # (B, T, d_model)
        x = self.norm(x)
        # Project to original feature space
        x = self.output_proj(x)                     # (B, T, C)
        return self.fcn(x)


# ==================== DCdetector ====================
# Core idea: Patch-based dual attention (patch-wise + channel-wise) with
# contrastive representation learning for anomaly detection.
class PatchEmbedding(nn.Module):
    """Split time series into overlapping patches and embed."""
    def __init__(self, n_window, patch_len, stride, n_feats, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (n_window - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * n_feats, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        patches = []
        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :].reshape(B, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)        # (B, num_patches, patch_len*C)
        out = self.embed(patches) + self.pos_embed   # (B, num_patches, d_model)
        return out


class DualAttentionBlock(nn.Module):
    """Dual attention: patch-wise temporal attention + channel-wise feature attention."""
    def __init__(self, d_model, n_feats, nhead=8, dropout=0.1):
        super(DualAttentionBlock, self).__init__()
        # Patch-wise self-attention
        self.patch_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Channel-wise: project d_model → n_feats space, then attend over features
        self.channel_proj = nn.Linear(d_model, n_feats)
        # Pick nhead that divides n_feats
        ch_nhead = 1
        for h in range(min(nhead, n_feats), 0, -1):
            if n_feats % h == 0:
                ch_nhead = h
                break
        self.channel_attn = nn.MultiheadAttention(n_feats, ch_nhead, dropout=dropout, batch_first=True)
        self.channel_proj_back = nn.Linear(n_feats, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # x: (B, num_patches, d_model)
        # 1. Patch-wise attention (temporal)
        attn_out, attn_weights = self.patch_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        # 2. Channel-wise attention (over features)
        B, N, D = x.shape
        x_proj = self.channel_proj(x)                # (B, num_patches, n_feats)
        x_t = x_proj.permute(0, 2, 1)                # (B, n_feats, num_patches)
        # Self-attention over feature channels
        ch_out, _ = self.channel_attn(x_proj, x_proj, x_proj)  # (B, num_patches, n_feats)
        x_ch = self.channel_proj_back(ch_out)        # (B, num_patches, d_model)
        x = self.norm2(x + self.dropout2(x_ch))
        # 3. Feed-forward
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return (x, attn_weights) if return_attention else x


class DCdetector(nn.Module):
    """DCdetector: Dual Attention Contrastive Representation Learning for
    Time Series Anomaly Detection.
    Patch-based dual attention (patch-wise + channel-wise) with contrastive
    two-view mechanism. Anomaly score = reconstruction error.
    """
    def __init__(self, feats):
        super(DCdetector, self).__init__()
        self.name = 'DCdetector'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.patch_len = 3
        self.stride = 2
        self.d_model = 64
        self.nhead = 8
        self.n_layers = 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            self.n_window, self.patch_len, self.stride, feats, self.d_model
        )
        self.num_patches = self.patch_embed.num_patches

        # Dual attention blocks
        self.blocks = nn.ModuleList([
            DualAttentionBlock(self.d_model, feats, self.nhead)
            for _ in range(self.n_layers)
        ])

        # Reconstruction head: d_model → patch_len * feats
        self.reconstruct = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.patch_len * feats),
        )
        self.fcn = nn.Sigmoid()

    def _fold_patches(self, rec_patches, B, T, C):
        """Reconstruct full time series by averaging overlapping patches."""
        rec = torch.zeros(B, T, C, device=rec_patches.device)
        cnt = torch.zeros(B, T, 1, device=rec_patches.device)
        for i in range(self.num_patches):
            start = i * self.stride
            rec[:, start:start + self.patch_len, :] += \
                rec_patches[:, i, :].reshape(B, self.patch_len, C)
            cnt[:, start:start + self.patch_len, :] += 1
        return rec / cnt.clamp(min=1)

    def forward(self, x, return_attn=False):
        # x: (B, T, C) where T=n_window
        B, T, C = x.shape

        # Patch embedding
        h = self.patch_embed(x)                     # (B, num_patches, d_model)

        # Dual attention blocks
        for block in self.blocks:
            h = block(h)

        # Reconstruct patches
        rec_patches = self.reconstruct(h)            # (B, num_patches, patch_len*C)

        # Fold back to time series
        rec = self._fold_patches(rec_patches, B, T, C)

        return self.fcn(rec)


class CNNLSTM_Mamba(nn.Module):
    def __init__(self, feats, hidden_size=64, num_layers=2, dropout=0.2):
        super(CNNLSTM_Mamba, self).__init__()

        self.name = 'CNNLSTM_Mamba'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10

        # CNN
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=self.n_window, out_channels=64, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # LSTM
        self.lstm = nn.LSTM(feats, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, feats)
        self.BN1 = nn.BatchNorm1d(self.n_window)
        self.relu1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(256, feats)
        self.BN2 = nn.BatchNorm1d(feats)
        self.relu2 = nn.LeakyReLU()

        self.mamba1 = Mamba(d_model=38, d_state=256, d_conv=2, expand=1)
        self.mamba2 = Mamba(d_model=128, d_state=256, d_conv=2, expand=1)
        self.mamba3 = Mamba(d_model=256, d_state=256, d_conv=2, expand=1)
        self.mamba4 = Mamba(d_model=1, d_state=256, d_conv=2, expand=1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        print('111', x.shape, x.float().dtype)
        # print(self.mamba1.in_proj.weight.dtype)
        # print(self.mamba1.hidden_states.dtype)

        x = self.mamba1(x.float())
        print('222', x.shape)
        # x = x.permute(1, 0, 2)
        # N = x.size(0)
        # # print('N', N)
        # # print('输入LSTM', x.shape)
        # x_lstm = self.lstm(x)[0]
        # # print('lstm处理后', x_lstm.shape)
        # x_lstm = self.relu1(self.BN1(self.fc1(x_lstm)))
        # # print('调整大小之后', x_lstm.shape)
        # x_lstm = x_lstm.mean(1)
        # # print('第一个维度平均', x_lstm.shape)
        # x_lstm = x_lstm.unsqueeze(1)
        # # print('增加一个维度', x_lstm.shape)
        # # print('输出LSTM', x_lstm.shape)
        #
        # # print('*****************************')
        # # print('输入CNN', x.shape)
        # x_cnn = self.conv_block1(x)
        # # print('经过第一个卷积块', x_cnn.shape)
        # x_cnn = self.conv_block2(x_cnn)
        # # print('经过第二个卷积块', x_cnn.shape)
        # x_cnn = self.conv_block3(x_cnn)
        # # print('经过第三个卷积块', x_cnn.shape)
        # x_cnn = torch.mean(x_cnn, 2)
        # # print('第二个维度平均', x_cnn.shape)
        # x_cnn = self.relu2(self.BN2(self.fc2(x_cnn)))
        # # print('调整大小', x_cnn.shape)
        # x_cnn = x_cnn.unsqueeze(1)
        # # print('增加一个维度', x_cnn.shape)
        # # print('输出CNN', x_cnn.shape)
        #
        # x = torch.cat([x_cnn, x_lstm], dim=1)
        # x = torch.mean(x, 1)
        # x = x.unsqueeze(1)
        # # print('CNNLSTM拼接后平均：', x.shape)
        #
        # x = x.permute(1, 0, 2)
        # # print('调整', x.shape)

        return x
