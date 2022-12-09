import torch
from utils import get_backbone, compute_prototypes

class PrototypicalNetworks(torch.nn.Module):
    def __init__(self, backbone_name, variant_depth, dropout):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = get_backbone(backbone_name, variant_depth, dropout)

    def forward(self,
                support_images,
                support_labels,
                query_images):
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        z_proto = compute_prototypes(z_support, support_labels)

        dists = torch.cdist(z_query, z_proto)

        scores = -dists
        return scores