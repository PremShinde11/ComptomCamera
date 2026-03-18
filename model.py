# =============================================================================
# model.py
# =============================================================================
# Neural network architecture for Compton camera source localisation.
#
# ARCHITECTURE OVERVIEW:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  INPUT: N raw Compton events, each described by 8 measured quantities   │
# │  (scatter_x/y/z, absorb_x/y/z, scatter_angle, electron_energy)         │
# └──────────────────────┬──────────────────────────────────────────────────┘
#                        │
#              ┌─────────▼─────────┐
#              │  EventEncoder     │   Linear projection: 8 → D_MODEL
#              │  (per-event MLP)  │   Applied independently to each event
#              └─────────┬─────────┘
#                        │  (N_events × D_MODEL)
#              ┌─────────▼─────────┐
#              │  Transformer      │   Multi-head self-attention × N_LAYERS
#              │  Encoder          │   Events attend to each other to isolate
#              │                   │   signal from background noise.
#              └─────────┬─────────┘
#                        │  (N_events × D_MODEL)
#              ┌─────────▼─────────┐
#              │  Global Average   │   Aggregate all event representations
#              │  Pool (Masked)    │   into a single scene-level latent vector
#              └────────┬┬─────────┘
#                       ││  (D_MODEL,)
#      ┌────────────────┼┼────────────────────────────────┐
#      │                │                                 │
# ┌────▼───────────┐ ┌──▼─────────────────────────┐ ┌─────▼───────────────┐
# │ CNN Spatial    │ │ Coordinate & Conf Head     │ │ Count Head          │
# │ Decoder        │ │                            │ │                     │
# │ Reshape latent │ │ Linear → ReLU → Dropout    │ │ Linear → ReLU       │
# │ ConvTrans ×4   │ │ Linear → ReLU → Dropout    │ │ Linear → (6,)       │
# │ Conv1x1 →      │ │ Linear → (MAX_SOURCES * 4) │ │                     │
# │ Sigmoid        │ │                            │ │ Predicts total      │
# │                │ │ Predicts (x, y, z, conf)   │ │ number of sources   │
# │ Outputs 3D     │ │ for up to 5 sources in     │ │ present (0 to 5)    │
# │ Spatial Heatmap│ │ normalised [-1,+1] space   │ │                     │
# └────────────────┘ └────────────────────────────┘ └─────────────────────┘
#
# THREE-HEAD DESIGN RATIONALE:
#   1. CNN Heatmap Head: Produces a 3D visual probability map of the scene.
#   2. Coordinate Head: Predicts exactly WHERE up to 5 distinct sources are
#      located (x, y, z) and outputs a "confidence" score for each prediction
#      slot, allowing the network to handle varying numbers of sources.
#   3. Count Head: Explicitly classifies HOW MANY sources (0-5) are in the
#      scene, used at inference to select the top valid coordinate predictions.
#
# DFG Compton Camera Project — University of Siegen
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.init as init

import config


# =============================================================================
# COMPONENT 1 — PER-EVENT ENCODER
# =============================================================================

class EventEncoder(nn.Module):
    """
    Projects each raw Compton event from 8 input features to D_MODEL dimensions.

    This is applied INDEPENDENTLY to every event in the scene — it is a
    shared feature extractor that maps the raw physics measurements into
    a richer representation space before the Transformer sees them.

    Architecture:
        Linear(8 → D_MODEL) → LayerNorm → ReLU → Linear(D_MODEL → D_MODEL)

    The LayerNorm after the first linear makes training more stable by
    normalising the activations before the non-linearity.
    """

    def __init__(
        self,
        n_input_features: int,
        d_model:          int,
        dropout:          float,
    ):
        """
        Parameters
        ----------
        n_input_features : int   — raw features per event (8 in our case)
        d_model          : int   — Transformer embedding dimension
        dropout          : float — dropout rate
        """
        super().__init__()

        self.projection = nn.Sequential(
            # First layer: raw features → embedding space
            nn.Linear(n_input_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Second layer: refine the embedding
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        events : torch.Tensor  shape (batch, n_events, n_input_features)

        Returns
        -------
        torch.Tensor  shape (batch, n_events, d_model)
            Each event represented as a D_MODEL-dimensional embedding.
        """
        return self.projection(events)


# =============================================================================
# COMPONENT 2 — TRANSFORMER ENCODER
# =============================================================================

class ComptonTransformerEncoder(nn.Module):
    """
    Multi-layer Transformer encoder that processes a SET of Compton events.

    WHY A TRANSFORMER?
        A scene contains N events, where N varies and order is arbitrary
        (a detector records events in time order, but for source localisation
        the time ordering is irrelevant). Transformers are designed for
        variable-length, permutation-invariant sets.

        Self-attention allows each event to attend to ALL other events.
        This is physically meaningful: an event that is CONSISTENT with many
        other events (i.e. all Compton cones intersect at the same point)
        will receive higher attention weights. The network learns to
        up-weight signal events and down-weight background (fake coincidences).

    Architecture:
        N_ENCODER_LAYERS of:
            Multi-Head Self-Attention (with padding mask)
            → Add & LayerNorm
            → Feed-Forward Network (Linear → ReLU → Linear)
            → Add & LayerNorm

        This is the standard pre-LN Transformer encoder block
        (as in "Attention is All You Need", Vaswani et al. 2017).
    """

    def __init__(
        self,
        d_model:          int,
        n_heads:          int,
        n_encoder_layers: int,
        dim_feedforward:  int,
        dropout:          float,
    ):
        """
        Parameters
        ----------
        d_model          : int   — embedding dimension (must be divisible by n_heads)
        n_heads          : int   — number of parallel attention heads
        n_encoder_layers : int   — number of stacked encoder layers
        dim_feedforward  : int   — inner dimension of FFN sublayer
        dropout          : float — dropout rate
        """
        super().__init__()

        # Build one Transformer encoder layer
        single_encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,   # input shape is (batch, seq, features)
            norm_first      = True,   # pre-LN is more stable than post-LN
        )

        # Stack N_ENCODER_LAYERS of the above
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = single_encoder_layer,
            num_layers    = n_encoder_layers,
        )

    def forward(
        self,
        event_embeddings: torch.Tensor,
        padding_mask:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        event_embeddings : torch.Tensor  shape (batch, n_events, d_model)
            Per-event embeddings from the EventEncoder.

        padding_mask : torch.Tensor  shape (batch, n_events)  dtype=bool
            True  → this position is PADDING (should be ignored by attention).
            False → this position is a real event.
            Passed to PyTorch's key_padding_mask argument.

        Returns
        -------
        torch.Tensor  shape (batch, n_events, d_model)
            Contextualised event representations — each event now encodes
            information about its relationship to all other events.
        """
        contextualised = self.transformer_encoder(
            src              = event_embeddings,
            src_key_padding_mask = padding_mask,
        )
        return contextualised


# =============================================================================
# COMPONENT 3 — GLOBAL AGGREGATOR
# =============================================================================

class MaskedGlobalAveragePool(nn.Module):
    """
    Aggregates N per-event representations into one scene-level latent vector.

    WHY MASKED AVERAGE POOLING?
        Simple approaches like sum or max can be fooled by padding tokens.
        We compute the mean ONLY over real events (mask == False), ignoring
        the padded positions completely.

        The result is a single D_MODEL-dimensional vector that summarises
        the entire scene of Compton events — this is the "brain" of the
        network that both output heads read from.
    """

    def forward(
        self,
        contextualised_events: torch.Tensor,
        padding_mask:          torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        contextualised_events : torch.Tensor  shape (batch, n_events, d_model)
        padding_mask          : torch.Tensor  shape (batch, n_events)  bool
            True = padding position (excluded from average).

        Returns
        -------
        torch.Tensor  shape (batch, d_model)
            One latent vector per scene.
        """
        # real_mask: True for real events, False for padding — opposite convention
        real_mask = ~padding_mask                          # (batch, n_events)

        # Zero out the padded positions before summing
        real_mask_expanded = real_mask.unsqueeze(-1).float()  # (batch, n_events, 1)
        masked_sum = (contextualised_events * real_mask_expanded).sum(dim=1)

        # Divide by the number of real events in each scene (avoid /0)
        n_real_events = real_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)

        latent_vector = masked_sum / n_real_events         # (batch, d_model)
        return latent_vector


# =============================================================================
# COMPONENT 4 — CNN SPATIAL DECODER (Heatmap Head)
# =============================================================================

class SpatialHeatmapDecoder(nn.Module):
    """
    Decodes the scene latent vector into a 2D spatial heatmap.

    This is the "image output" of the network — it produces a picture showing
    where the source is located in the XY plane, which is what the user
    sees as the medical image result.

    Architecture:
        Latent vector (D_MODEL,)
            → Linear → reshape to (D_MODEL, 4, 4)   [tiny feature map]
            → ConvTranspose2d → (128, 8,  8 )        [upsample ×2]
            → ConvTranspose2d → ( 64, 16, 16)        [upsample ×2]
            → ConvTranspose2d → ( 32, 32, 32)        [upsample ×2]
            → ConvTranspose2d → ( 16, 64, 64)        [upsample ×2]
            → Conv2d(1×1)     → (  1, 64, 64)        [project to 1 channel]
            → Sigmoid                                [output in [0, 1]]

    Four upsampling stages: 4 → 8 → 16 → 32 → 64 = config.HEATMAP_SIZE.
    Each ConvTranspose block uses BatchNorm and ReLU for training stability.

    WHY LEARNED DECODING vs GEOMETRIC BACKPROJECTION?
        Geometric backprojection draws circles on an image for each event —
        it produces ring artifacts and treats all events equally (signal and
        background). The learned decoder uses the Transformer's output, which
        has already up-weighted consistent (signal) events and down-weighted
        inconsistent (background) events. The result is a cleaner image.
    """

    def __init__(
        self,
        d_model:            int,
        latent_spatial_dim: int,
        decoder_channels:   list,
        heatmap_size:       int,
    ):
        """
        Parameters
        ----------
        d_model            : int        — latent vector dimension (e.g. 128)
        latent_spatial_dim : int        — initial spatial size before upsampling (e.g. 4)
        decoder_channels   : list[int]  — channels at each upsampling stage
        heatmap_size       : int        — target output spatial size (e.g. 64)
        """
        super().__init__()

        self.d_model            = d_model
        self.latent_spatial_dim = latent_spatial_dim

        # ── Linear layer to expand latent vector into a small feature map ──
        # Output size: D_MODEL × latent_spatial_dim × latent_spatial_dim
        n_latent_pixels = d_model * latent_spatial_dim * latent_spatial_dim
        self.latent_to_feature_map = nn.Sequential(
            nn.Linear(d_model, n_latent_pixels),
            nn.ReLU(inplace=True),
        )

        # ── Upsampling blocks: each doubles spatial resolution ──────────────
        # We build these dynamically from decoder_channels.
        # In-channels for stage i = out-channels of stage i-1 (or D_MODEL for stage 0)
        upsample_blocks = []
        in_channels = d_model

        for out_channels in decoder_channels:
            upsample_blocks.append(
                self._make_upsample_block(in_channels, out_channels)
            )
            in_channels = out_channels

        self.upsample_blocks = nn.Sequential(*upsample_blocks)

        # ── Final 1×1 convolution: project to single-channel heatmap ────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], config.HEATMAP_Z_BINS, kernel_size=1),
            nn.Sigmoid(),   # output in [0, 1] — interpreted as probability
        )

    @staticmethod
    def _make_upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        One upsampling block: ConvTranspose2d (stride=2) + BatchNorm + ReLU.

        ConvTranspose2d with kernel=2, stride=2 doubles the spatial dimensions:
            (C, H, W) → (out_channels, 2H, 2W)
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2,    # doubles H and W
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_vector : torch.Tensor  shape (batch, d_model)

        Returns
        -------
        torch.Tensor  shape (batch, HEATMAP_Z_BINS, heatmap_size, heatmap_size)
            Spatial heatmap — high values indicate likely source locations.
        """
        batch_size = latent_vector.shape[0]

        # Expand latent vector → small 3D feature map
        feature_map = self.latent_to_feature_map(latent_vector)  # (batch, d_model*4*4)
        feature_map = feature_map.view(
            batch_size,
            self.d_model,
            self.latent_spatial_dim,
            self.latent_spatial_dim,
        )                                                          # (batch, D, 4, 4)

        # Progressive upsampling
        upsampled = self.upsample_blocks(feature_map)             # (batch, 16, 64, 64)

        # Final projection to single-channel heatmap
        heatmap = self.final_conv(upsampled)                      # (batch, 1, 64, 64)

        return heatmap


# =============================================================================
# COMPONENT 5 — COORDINATE MLP HEAD (Direct Prediction Head)
# =============================================================================

class CoordinateMlpHead(nn.Module):
    """
    Predicts source (x, y, z) directly from the scene latent vector.

    This head produces exact 3D coordinates (normalised to [-1, +1]) rather
    than an image. It is used to compute mm-accuracy error metrics (err_x,
    err_y, err_z, err_xyz) and to directly supervise the network on the
    coordinate regression task.

    Architecture:
        Latent (D_MODEL,)
            → Linear(D_MODEL → D_MODEL) → LayerNorm → ReLU → Dropout
            → Linear(D_MODEL → D_MODEL // 2) → LayerNorm → ReLU → Dropout
            → Linear(D_MODEL // 2 → 3)
            → Tanh   (output in [-1, +1])

    Tanh is used instead of Sigmoid because target coordinates are
    normalised to [-1, +1] (not [0, 1]).
    """

    def __init__(self, d_model: int, dropout: float, n_outputs: int = 3):
        """
        Parameters
        ----------
        d_model   : int   — latent vector dimension
        dropout   : float — dropout rate
        n_outputs : int   — number of output coordinates (3 for x, y, z)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            # Layer 1: full-width
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Layer 2: compress to half-width
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Output layer: predict (x, y, z, conf_logit) in raw space
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent_vector : torch.Tensor  shape (batch, d_model)

        Returns
        -------
        torch.Tensor
            Predicted (x, y, z) in normalised [-1, +1] space and confidence logits.
            Denormalise using config.TARGET_BOUNDS to get mm values.
        """
        output = self.mlp(latent_vector)
        if output.shape[-1] == 3:
            return torch.tanh(output)
        if output.shape[-1] % 4 == 0:
            batch_size = output.shape[0]
            reshaped = output.view(batch_size, -1, 4)
            coords = torch.tanh(reshaped[:, :, :3])
            conf = reshaped[:, :, 3:4]
            return torch.cat([coords, conf], dim=-1).view(batch_size, -1)
        return output


# =============================================================================
# COMPONENT 6 — COUNT PREDICTION HEAD
# =============================================================================

class CountHead(nn.Module):
    """
    Predicts the number of sources in the scene (0–5).

    This is a classification head that outputs raw logits for each possible
    source count. The predicted count is used at inference to select how
    many predictions to keep from the coordinate head.

    Architecture:
        Latent (D_MODEL,)
            → Linear(D_MODEL → D_MODEL // 2) → ReLU
            → Linear(D_MODEL // 2 → max_sources + 1)  # logits for 0,1,2,3,4,5
    """

    def __init__(self, d_model: int, max_sources: int = 5):
        """
        Parameters
        ----------
        d_model       : int   — latent vector dimension
        max_sources   : int   — maximum number of sources (default 5)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, max_sources + 1),  # 6 outputs: 0,1,2,3,4,5
        )

        # Initialize ALL layers with small weights for stable count prediction start
        # Hidden layer: small Kaiming init
        init.kaiming_uniform_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        init.constant_(self.mlp[0].bias, 0.0)
        
        # Output layer: near-zero init so logits start uniform across all classes
        # This ensures cross_entropy ≈ log(6) = 1.79 at initialization
        init.normal_(self.mlp[2].weight, mean=0.0, std=0.01)
        init.constant_(self.mlp[2].bias, 0.0)

    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:

        return self.mlp(latent_vector)  # (batch, 6)


# =============================================================================
# FULL MODEL — ComptonSourceLocaliser
# =============================================================================

class ComptonSourceLocaliser(nn.Module):
    """
    Full end-to-end model for Compton camera source localisation.

    Combines all five components into one forward pass:
        1. EventEncoder          — per-event feature projection
        2. TransformerEncoder    — cross-event self-attention
        3. MaskedGlobalAvgPool  — aggregate N events → 1 latent vector
        4. SpatialHeatmapDecoder — latent → 2D XY image (for visualisation)
        5. CoordinateMlpHead    — latent → (x, y, z) coordinates (for metrics)

    Input  : (batch, n_events, 8)  + (batch, n_events) padding mask
    Output : heatmap (batch, 1, 64, 64)  +  coords (batch, 3)
    """

    def __init__(self):
        super().__init__()

        # ── Component 1: per-event encoder ────────────────────────────────
        self.event_encoder = EventEncoder(
            n_input_features = config.N_INPUT_FEATURES,
            d_model          = config.D_MODEL,
            dropout          = config.DROPOUT,
        )

        # ── Component 2: Transformer encoder ──────────────────────────────
        self.transformer_encoder = ComptonTransformerEncoder(
            d_model          = config.D_MODEL,
            n_heads          = config.N_HEADS,
            n_encoder_layers = config.N_ENCODER_LAYERS,
            dim_feedforward  = config.DIM_FEEDFORWARD,
            dropout          = config.DROPOUT,
        )

        # ── Component 3: masked global average pooling ─────────────────────
        self.global_pool = MaskedGlobalAveragePool()

        # ── Component 4: CNN spatial decoder (heatmap head) ───────────────
        self.heatmap_decoder = SpatialHeatmapDecoder(
            d_model            = config.D_MODEL,
            latent_spatial_dim = config.LATENT_SPATIAL_DIM,
            decoder_channels   = config.DECODER_CHANNELS,
            heatmap_size       = config.HEATMAP_SIZE,
        )

        # ── Component 5: coordinate MLP head ──────────────────────────────
        # FIX: predict MAX_SOURCES*4 coordinates (x,y,z,confidence), not just 3.
        # 5 sources x 4 coords = 20 outputs. Reshaped to (B, MAX_SOURCES, 4) in forward().
        self._max_sources = config.MAX_SOURCES   # must match MAX_SOURCES in train.py
        self.coordinate_head = CoordinateMlpHead(
            d_model   = config.D_MODEL,
            dropout   = config.DROPOUT,
            n_outputs = self._max_sources * 4,   # 4 = x, y, z, confidence logit
        )

        # ── Component 6: count prediction head ────────────────────────────
        self.count_head = CountHead(
            d_model       = config.D_MODEL,
            max_sources   = config.MAX_SOURCES,
        )

        # ── Weight initialisation ──────────────────────────────────────────
        self._initialise_weights()

    def _initialise_weights(self):
        """
        Applies sensible weight initialisation throughout the network.

        Linear layers: Kaiming (He) uniform init — designed for ReLU networks.
        LayerNorm:     standard (weight=1, bias=0).
        ConvTranspose: Kaiming uniform.
        BatchNorm:     standard (weight=1, bias=0).
        
        EXCLUSION: Count head final layer is initialized in CountHead.__init__
        with near-zero weights for stable cross-entropy start (~1.79).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Skip count head final layer - already initialized properly
                if name == 'count_head.mlp.2':
                    continue
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        events:       torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the entire model.

        Parameters
        ----------
        events : torch.Tensor  shape (batch, n_events, N_INPUT_FEATURES)
            Raw measured event features, normalised to [-1, +1].

        padding_mask : torch.Tensor  shape (batch, n_events)  dtype=bool
            True  = padding position (ignored by attention and pooling).
            False = real event.

        Returns
        -------
        heatmap : torch.Tensor  shape (batch, HEATMAP_Z_BINS, HEATMAP_SIZE, HEATMAP_SIZE)
            Spatial probability map for XY source location.
            Values in [0, 1] from Sigmoid activation.

        coords_and_conf : torch.Tensor  shape (batch, MAX_SOURCES, 4)
            Predicted (x, y, z, confidence_logit) in normalised space.
            x,y,z in [-1, +1] from Tanh activation.
            confidence_logit is raw logit (apply sigmoid for probability).

        count_logits : torch.Tensor  shape (batch, MAX_SOURCES + 1)
            Raw logits for source count classification (0 to MAX_SOURCES).
        """
        # ── Step 1: Project each event to D_MODEL dimensions ──────────────
        # Shape: (batch, n_events, 8) → (batch, n_events, D_MODEL)
        event_embeddings = self.event_encoder(events)

        # ── Step 2: Self-attention — events contextualise each other ───────
        # Shape: (batch, n_events, D_MODEL) → same shape
        contextualised_events = self.transformer_encoder(
            event_embeddings, padding_mask
        )

        # ── Step 3: Aggregate all events into one scene latent vector ──────
        # Shape: (batch, n_events, D_MODEL) → (batch, D_MODEL)
        scene_latent = self.global_pool(contextualised_events, padding_mask)

        # ── Step 4: Decode latent vector into spatial heatmap (image) ──────
        # Shape: (batch, D_MODEL) → (batch, 1, 64, 64)
        heatmap = self.heatmap_decoder(scene_latent)

        # ── Step 5: Predict all source coordinates + confidence ────────────
        # Shape: (batch, D_MODEL) → (batch, MAX_SOURCES*4) → (batch, MAX_SOURCES, 4)
        flat_coords_and_conf = self.coordinate_head(scene_latent)
        coords_and_conf = flat_coords_and_conf.view(
            flat_coords_and_conf.shape[0], self._max_sources, 4
        )

        # ── Step 6: Predict source count ───────────────────────────────────
        # Shape: (batch, D_MODEL) → (batch, MAX_SOURCES + 1)
        count_logits = self.count_head(scene_latent)

        return heatmap, coords_and_conf, count_logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class ComptonLocalisationLoss(nn.Module):
    """
    Combined loss for the three-head architecture.

    Total loss = lambda_heatmap * L_heatmap + lambda_coord * L_coord 
               + lambda_confidence * L_confidence + lambda_count * L_count

    L_heatmap:
        Binary cross-entropy between predicted heatmap and target Gaussian blob.
        BCE is better than MSE for heatmaps because the target is a probability
        distribution — BCE penalises confidently wrong predictions more.

    L_coord:
        Mean squared error between predicted and true (x, y, z) coordinates
        in normalised space. Divided by 3 so each axis contributes equally.

    L_confidence:
        Binary cross-entropy for confidence prediction. A prediction slot gets
        label=1 if it was matched to a real source, label=0 otherwise.

    L_count:
        Cross-entropy loss for source count classification (0-5 sources).

    The losses are added with configurable weights (lambda_*) so you
    can easily tune the trade-off.
    """

    def __init__(
        self,
        lambda_heatmap: float = config.LAMBDA_HEATMAP,
        lambda_coord:   float = config.LAMBDA_COORD,
        lambda_confidence: float = config.LAMBDA_CONFIDENCE,
        lambda_count:   float = config.LAMBDA_COUNT,
    ):
        """
        Parameters
        ----------
        lambda_heatmap   : float — weight for the heatmap BCE loss
        lambda_coord     : float — weight for the coordinate MSE loss
        lambda_confidence: float — weight for confidence BCE loss
        lambda_count     : float — weight for count cross-entropy loss
        """
        super().__init__()

        self.lambda_heatmap    = lambda_heatmap
        self.lambda_coord      = lambda_coord
        self.lambda_confidence = lambda_confidence
        self.lambda_count      = lambda_count

        # BCE with logits is numerically more stable than Sigmoid + BCE,
        # but our model outputs Sigmoid already so we use standard BCE.
        self.bce_loss = nn.BCELoss()
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()  # for count classification

    def forward(
        self,
        pred_heatmap:  torch.Tensor,
        target_heatmap: torch.Tensor,
        pred_coords_and_conf: torch.Tensor,
        target_coords: torch.Tensor,
        confidence_labels: torch.Tensor,
        count_logits: torch.Tensor,
        count_targets: torch.LongTensor,
        matched_slot_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the combined loss and its components.

        Parameters
        ----------
        pred_heatmap      : (batch, 1, H, W)  — predicted spatial heatmap
        target_heatmap    : (batch, 1, H, W)  — target Gaussian blob heatmap
        pred_coords_and_conf: (batch, MAX_SOURCES, 4) — predicted (x,y,z,confidence_logit)
        target_coords     : (batch, MAX_SOURCES, 3)   — true (x,y,z) normalised, NaN for absent
        confidence_labels : (batch, MAX_SOURCES)      — 1 if matched to real source, 0 otherwise
        count_logits      : (batch, MAX_SOURCES + 1)  — raw logits for count 0..MAX_SOURCES
        count_targets     : (batch,)                — true source count per scene (0..MAX_SOURCES)
        matched_slot_mask : (batch, MAX_SOURCES)      — optional, True for matched prediction slots

        Returns
        -------
        total_loss        : scalar tensor — weighted sum of all losses
        loss_heatmap      : scalar tensor — heatmap component (for logging)
        loss_coord        : scalar tensor — coordinate component (for logging)
        loss_confidence   : scalar tensor — confidence component (for logging)
        loss_count        : scalar tensor — count component (for logging)
        """
        # Heatmap loss: BCE between predicted map and Gaussian target
        loss_heatmap = self.bce_loss(pred_heatmap, target_heatmap)

        # Coordinate loss: MSE on normalised (x, y, z) predictions
        # Only compute loss where target is not NaN (valid sources)
        valid_mask = ~torch.isnan(target_coords).any(dim=-1)  # (batch, MAX_SOURCES)
        if valid_mask.any():
            # Mask out invalid entries
            pred_valid = pred_coords_and_conf[:, :, :3].clone()
            target_valid = target_coords.clone()
            pred_valid[~valid_mask] = 0
            target_valid[~valid_mask] = 0
            
            # Compute MSE only over valid sources
            n_valid = valid_mask.sum().float().clamp(min=1.0)
            coord_weights = torch.tensor([1.0, 1.0, 4.0], device=pred_coords_and_conf.device)
            loss_coord = ((pred_valid - target_valid) ** 2 * coord_weights).sum(dim=-1)
            loss_coord = (loss_coord * valid_mask.float()).sum() / n_valid
        else:
            loss_coord = torch.tensor(0.0, device=pred_coords_and_conf.device)

        # Confidence loss: BCE between predicted confidence and matching labels
        # pred_coords_and_conf[:, :, 3] contains raw confidence logits
        pred_conf_logits = pred_coords_and_conf[:, :, 3]  # (batch, MAX_SOURCES)
        loss_confidence = self.bce_logits_loss(pred_conf_logits, confidence_labels)

        # Count loss: cross-entropy between predicted logits and true counts
        loss_count = self.ce_loss(count_logits, count_targets)
        
        # FIX Issue 4: Regularization for unmatched slots (ghost predictions)
        # Push unmatched prediction slots toward origin (0,0,0) to prevent drift
        loss_regularise = torch.tensor(0.0, device=pred_coords_and_conf.device)
        if matched_slot_mask is not None:
            unmatched_mask = ~matched_slot_mask  # True for unmatched slots
            if unmatched_mask.any():
                # Get coordinates of unmatched slots
                unmatched_preds = pred_coords_and_conf[:, :, :3][unmatched_mask]
                # Penalize distance from origin
                loss_regularise = (unmatched_preds ** 2).mean()

        # Weighted combination
        total_loss = (
            self.lambda_heatmap    * loss_heatmap +
            self.lambda_coord      * loss_coord +
            self.lambda_confidence * loss_confidence +
            self.lambda_count      * loss_count +
            config.LAMBDA_REGULARISE * loss_regularise  # use config value for regularization
        )

        return total_loss, loss_heatmap, loss_coord, loss_confidence, loss_count


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    """
    Runs a quick forward pass with random tensors to verify:
      - All tensor shapes are correct throughout the model.
      - The loss computes without errors.
      - Parameter count is reasonable.
    """
    print("=" * 60)
    print("  ComptonSourceLocaliser — shape sanity check")
    print("=" * 60)

    device = config.DEVICE
    print(f"  Device: {device}")

    # Simulate a batch of 4 scenes, each with 1000 events, 8 features
    batch_size = 4
    n_events   = config.MAX_EVENTS_PER_SCENE

    # Random input events (normalised to [-1, +1] as dataset.py will produce)
    dummy_events = torch.randn(batch_size, n_events, config.N_INPUT_FEATURES)

    # Padding mask: last 200 positions are padding (True = ignore)
    dummy_padding_mask = torch.zeros(batch_size, n_events, dtype=torch.bool)
    dummy_padding_mask[:, 800:] = True   # positions 800-999 are padding

    # Target heatmap: Gaussian blob at a random location
    dummy_target_heatmap = torch.zeros(
        batch_size,
        config.HEATMAP_Z_BINS,
        config.HEATMAP_SIZE,
        config.HEATMAP_SIZE,
    )
    z_mid = int(config.HEATMAP_Z_BINS // 2)
    dummy_target_heatmap[:, z_mid, 32, 32] = 1.0   # source at centre

    # Target coordinates (normalised)
    dummy_target_coords = torch.zeros(batch_size, config.N_TARGET_FEATURES)

    # Build model and loss
    model = ComptonSourceLocaliser().to(device)
    criterion = ComptonLocalisationLoss()

    dummy_events       = dummy_events.to(device)
    dummy_padding_mask = dummy_padding_mask.to(device)
    dummy_target_heatmap = dummy_target_heatmap.to(device)
    dummy_target_coords  = dummy_target_coords.to(device)

    # Forward pass
    pred_heatmap, pred_coords = model(dummy_events, dummy_padding_mask)

    # Loss
    total_loss, loss_h, loss_c = criterion(
        pred_heatmap, dummy_target_heatmap,
        pred_coords,  dummy_target_coords,
    )

    print(f"\n  Input events shape    : {tuple(dummy_events.shape)}")
    print(f"  Padding mask shape    : {tuple(dummy_padding_mask.shape)}")
    print(f"  Output heatmap shape  : {tuple(pred_heatmap.shape)}")
    print(f"  Output coords shape   : {tuple(pred_coords.shape)}")
    print(f"\n  Total loss            : {total_loss.item():.4f}")
    print(f"  Heatmap loss          : {loss_h.item():.4f}")
    print(f"  Coordinate loss       : {loss_c.item():.4f}")
    print(f"\n  Total parameters      : {model.count_parameters():,}")
    print("=" * 60)
    print("  All shapes correct. Model ready for training.")
    print("=" * 60)
