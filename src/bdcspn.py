import torch
from utils import get_backbone, compute_prototypes, rectify_prototypes, cosine_distance_to_prototypes, softmax_if_specified

class BDCosineSimilarityProtonet(torch.nn.Module):
    def __init__(self, backbone_name, variant_depth, dropout, use_softmax=False):
        super(BDCosineSimilarityProtonet, self).__init__()
        self.backbone = get_backbone(backbone_name, variant_depth, dropout)
        self.use_softmax = use_softmax

    def forward(self,
                support_images,
                support_labels,
                query_images):
        """
        Predict query labels using labeled support images.
        """
        z_support = self.backbone.forward(support_images)
        z_proto = compute_prototypes(z_support, support_labels)

        z_query = self.backbone.forward(query_images)

        z_proto = rectify_prototypes(
            prototypes=z_proto, support_features=z_support, support_labels=support_labels, query_features=z_query
        )
        
        dists = cosine_distance_to_prototypes(z_query, z_proto)
        
        return softmax_if_specified(dists) if self.use_softmax else dists

    @staticmethod
    def is_transductive():
        return True