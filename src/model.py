import torch
import torch.nn as nn

class DualHeadVAEGenerator(nn.Module):
    def __init__(self, num_classes=4, latent_dim=128,
                 class_emb_dim=32, class_mlp_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # ------------------- Encoder (geometry + class) -------------------
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_fc_geom = nn.Linear(256 * 8 * 8, 512)

        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(class_emb_dim, class_mlp_dim),
            nn.ReLU(inplace=True)
        )

        fusion_dim = 512 + class_mlp_dim
        self.fc_mu     = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)

        # ------------------- Decoder -------------------
        dec_input_dim = latent_dim + class_mlp_dim
        self.dec_fc = nn.Linear(dec_input_dim, 256 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def decode(self, z, c_feat):
        h = self.dec_fc(torch.cat([z, c_feat], dim=1))
        h = h.view(h.size(0), 256, 8, 8)
        return self.dec_conv(h)
