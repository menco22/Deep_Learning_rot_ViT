from utils.functions import *
from utils.imports import *
from models.ViT import *

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads,
                                                    batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
          nn.Linear(hidden_size, hidden_size * 2),
          nn.GELU(),
          nn.Dropout(0.1),
          nn.Linear(hidden_size * 2, hidden_size),
          nn.Dropout(0.1)
          )

    def forward(self, x):
        norm_x = self.norm1(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x
        norm_x = self.norm2(x)
        x = self.mlp(norm_x) + x
        return x

class FeatAnalysisViT(nn.Module):
    def __init__(self, image_size, channels_in, patch_size, hidden_size, num_classes, num_layers, num_heads):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_size, num_classes)
        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std=0.001))
        self.cls_pos_embedding = nn.Parameter(torch.empty(1, 1, hidden_size).normal_(std=0.001))

    def forward(self, image, return_intermediates=False, layer_index=None):
        bs = image.shape[0]
        patch_seq = extract_patches(image, patch_size=self.patch_size)
        patch_emb = self.fc_in(patch_seq)
        embs = torch.cat((self.out_vec.expand(bs, 1, -1), patch_emb), 1)
        embs = embs + torch.cat((self.cls_pos_embedding, self.pos_embedding), dim=1)

        if layer_index is not None:
            if layer_index == 0:
                return embs[:, 0]

            for i in range(min(layer_index, len(self.blocks))):
                embs = self.blocks[i](embs)

            return embs[:, 0]
        else:
            for block in self.blocks:
                embs = block(embs)
            return self.fc_out(embs[:, 0])
        
class FeatureClassifier(nn.Module):
    def __init__(self, backbone, layer_index, num_classes):
        super(FeatureClassifier, self).__init__()
        self.backbone = backbone
        self.layer_index = layer_index
        self.classifier = nn.Linear(backbone.hidden_size, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x, layer_index=self.layer_index)
        return self.classifier(features)