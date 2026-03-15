# Quick test of new architecture
import torch
import config
from model import ComptonSourceLocaliser

print("=" * 60)
print("  Testing New Architecture with Count + Confidence")
print("=" * 60)

device = config.DEVICE
print(f"\n  Device: {device}")
print(f"  MAX_SOURCES: {config.MAX_SOURCES}")
print(f"  LAMBDA_COUNT: {config.LAMBDA_COUNT}")
print(f"  LAMBDA_CONFIDENCE: {config.LAMBDA_CONFIDENCE}")

# Create dummy input
batch_size = 2
n_events = config.MAX_EVENTS_PER_SCENE

dummy_events = torch.randn(batch_size, n_events, config.N_INPUT_FEATURES)
dummy_padding_mask = torch.zeros(batch_size, n_events, dtype=torch.bool)
dummy_padding_mask[:, 800:] = True  # last 200 are padding

# Target tensors
dummy_target_heatmap = torch.zeros(batch_size, 1, config.HEATMAP_SIZE, config.HEATMAP_SIZE)
dummy_target_heatmap[:, :, 32, 32] = 1.0

dummy_target_coords = torch.full((batch_size, config.MAX_SOURCES, 3), float('nan'))
dummy_target_coords[:, 0, :] = 0.0  # First source at origin
dummy_confidence_labels = torch.ones(batch_size, config.MAX_SOURCES)
dummy_count_targets = torch.tensor([1, 2])  # Scene 1 has 1 source, scene 2 has 2

# Build model
model = ComptonSourceLocaliser().to(device)
print(f"\n  Model parameters: {model.count_parameters():,}")

# Forward pass
dummy_events = dummy_events.to(device)
dummy_padding_mask = dummy_padding_mask.to(device)

print("\n  Running forward pass...")
heatmap, coords_and_conf, count_logits = model(dummy_events, dummy_padding_mask)

print(f"  Heatmap shape:      {tuple(heatmap.shape)}")
print(f"  Coords+Conf shape:  {tuple(coords_and_conf.shape)}")
print(f"  Count logits shape: {tuple(count_logits.shape)}")

# Verify shapes
assert heatmap.shape == (batch_size, 1, config.HEATMAP_SIZE, config.HEATMAP_SIZE)
assert coords_and_conf.shape == (batch_size, config.MAX_SOURCES, 4)
assert count_logits.shape == (batch_size, config.MAX_SOURCES + 1)

print("\n  ✓ All shapes correct!")

# Test loss
from model import ComptonLocalisationLoss

dummy_target_coords = dummy_target_coords.to(device)
dummy_confidence_labels = dummy_confidence_labels.to(device)
dummy_count_targets = dummy_count_targets.to(device)
dummy_target_heatmap = dummy_target_heatmap.to(device)

criterion = ComptonLocalisationLoss()

print("\n  Computing loss...")
total_loss, l_heatmap, l_coord, l_conf, l_count = criterion(
    heatmap, dummy_target_heatmap,
    coords_and_conf, dummy_target_coords,
    dummy_confidence_labels, count_logits, dummy_count_targets
)

print(f"  Total loss:     {total_loss.item():.4f}")
print(f"  Heatmap loss:   {l_heatmap.item():.4f}")
print(f"  Coord loss:     {l_coord.item():.4f}")
print(f"  Conf loss:      {l_conf.item():.4f}")
print(f"  Count loss:     {l_count.item():.4f}")

print("\n" + "=" * 60)
print("  ✓ Architecture test PASSED!")
print("=" * 60)
