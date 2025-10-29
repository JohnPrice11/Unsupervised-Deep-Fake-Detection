"""
    Modified Model Code using timm Xception, SwinV2, and ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import timm
import torch.utils.checkpoint # Included if checkpointing is ever used
from collections import OrderedDict # Needed for state dict handling


class ECL(nn.Module):
    """
    Enhanced Contrastive Learner (ECL) model.
    Consists of a backbone (timm Xception, SwinV2, or ConvNeXt) and a projection head.
    """
    def __init__(self, out_dim=128, backbone_type='xception'):
        super(ECL, self).__init__()
        self.backbone_type = backbone_type
        self.out_dim = out_dim

        # Define common projection head parameters (can be tuned)
        dim_mlp_hidden = 2048 # As per description/common practice for projection head
        dim_mlp_in = None # To be determined based on backbone output features

        # --- Backbone Selection ---
        if backbone_type == 'swinv2':
            # Create the SwinV2 backbone without its default head using timm
            try:
                # Using a powerful pre-trained model, e.g., on ImageNet-22k then finetuned on ImageNet-1k
                self.backbone = timm.create_model(
                    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
                    pretrained=True, # Load standard timm pretrained weights
                    num_classes=0 # Get features *before* the default head (Global Avg Pool output)
                )
                # Get the feature dimension from the timm model's attribute
                dim_mlp_in = self.backbone.num_features
                logging.info(f"Created timm SwinV2 backbone with feature dimension: {dim_mlp_in}")
            except Exception as e:
                logging.error(f"Failed to create timm SwinV2 model: {e}")
                raise ValueError(f"Could not create timm SwinV2 backbone. Make sure timm is installed and the model name is correct. Error: {e}")

        elif backbone_type == 'xception':
            # Create the Xception backbone using timm
            try:
                # Load standard timm pretrained weights from ImageNet-1k
                self.backbone = timm.create_model(
                    'xception',
                    pretrained=True, # Load standard timm pretrained weights
                    num_classes=0 # Get features *before* the default head (Global Avg Pool output)
                )
                 # Get the feature dimension from the timm model's attribute
                dim_mlp_in = self.backbone.num_features
                logging.info(f"Created timm Xception backbone with feature dimension: {dim_mlp_in}")
            except Exception as e:
                logging.error(f"Failed to create timm Xception model: {e}")
                raise ValueError(f"Could not create timm Xception backbone. Make sure timm is installed and the model name 'xception' is correct. Error: {e}")

        # --- Add ConvNeXt Backbone ---
        elif backbone_type == 'convnext_base':
            # Create the ConvNeXt Base backbone using timm
            try:
                # Using a powerful pre-trained model, e.g., on ImageNet-22k then finetuned on ImageNet-1k
                # This is a common and strong variant
                self.backbone = timm.create_model(
                    'convnext_base.fb_in22k_ft_in1k',
                    pretrained=True, # Load standard timm pretrained weights
                    num_classes=0 # Get features *before* the default head (Global Avg Pool output)
                )
                 # Get the feature dimension from the timm model's attribute
                dim_mlp_in = self.backbone.num_features
                logging.info(f"Created timm ConvNeXt Base backbone with feature dimension: {dim_mlp_in}")
            except Exception as e:
                logging.error(f"Failed to create timm ConvNeXt Base model: {e}")
                raise ValueError(f"Could not create timm ConvNeXt Base backbone. Make sure timm is installed and the model name is correct. Error: {e}")

        else:
            # Raise error if backbone type is not recognized
            raise ValueError(f"Unsupported backbone: {backbone_type}. Supported: 'xception', 'swinv2', 'convnext_base'")

        # --- Projection Head Definition (uses dim_mlp_in determined by backbone) ---
        if dim_mlp_in is None:
             raise ValueError("Backbone feature dimension (dim_mlp_in) was not determined.")

        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp_in, dim_mlp_hidden),
            nn.ReLU(inplace=False), # Keeping inplace=False as per your original code (safer for some debugging)
            nn.Linear(dim_mlp_hidden, out_dim) # Output size is out_dim (usually 128)
        )


        # Input normalization parameters (ImageNet mean/std)
        # Register as buffers so they are saved/loaded with the model state dict
        # These are typically used by the transforms in your data loader, not directly here
        # unless you explicitly add input normalization in the forward pass.
        self.register_buffer('input_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        logging.info(f"Initialized ECL model - {backbone_type}")


    def forward(self, x):
        """
        Forward pass for ECL.
        Applies backbone and then projection head.
        Returns features before projection (feat) and normalized features after projection (feat_contrast).
        """
        # Apply backbone forward pass
        if self.backbone_type in ['xception', 'swinv2', 'convnext_base']:
             # timm models with num_classes=0 return features after global pooling
             feat = self.backbone(x)
        else:
             # This should be caught by __init__, but included for safety
             raise ValueError(f"Unsupported backbone in forward: {self.backbone_type}")

        # Ensure features are flat [batch_size, feature_dim]
        # timm models with num_classes=0 usually output this directly,
        # but adding .flatten(1) makes it robust if a model outputs >2D tensor
        # e.g., [B, C, 1, 1] after global average pooling
        feat = feat.flatten(1)


        # Pass features through the projection head to get low-dimensional features (z)
        feat_z = self.projection_head(feat)

        # Normalize z to get feat_contrast (z_norm) for contrastive loss
        feat_contrast = F.normalize(feat_z, dim=1)

        # Return both f (features before projection) and z_norm (features after projection, normalized)
        # Your loss function (ECLoss) expects feat_contrast
        # You might use feat for other purposes if needed (e.g., linear evaluation head)
        return feat, feat_contrast