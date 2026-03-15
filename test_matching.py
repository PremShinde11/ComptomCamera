# Test matching-based confidence label generation
import torch
import numpy as np
from train import compute_minimum_cost_matching, build_confidence_labels_from_matching, MAX_SOURCES

print("=" * 60)
print("  Testing Matching-Based Confidence Labels")
print("=" * 60)

# Test case 1: Simple 2-source scene
print("\n  Test 1: Scene with 2 true sources")
batch_size = 1
pred_coords = torch.zeros(1, MAX_SOURCES, 3)
pred_coords[0, 0, :] = torch.tensor([0.1, 0.2, 0.3])  # pred slot 0
pred_coords[0, 1, :] = torch.tensor([-0.1, 0.5, 0.2]) # pred slot 1
pred_coords[0, 2, :] = torch.tensor([0.8, 0.9, 0.7])  # pred slot 3 (far away)
pred_coords[0, 3, :] = torch.tensor([-0.5, -0.5, 0.1]) # pred slot 4 (far away)
pred_coords[0, 4, :] = torch.tensor([0.0, 0.0, 0.0])  # pred slot 5 (center)

target_coords = torch.full((1, MAX_SOURCES, 3), float('nan'))
target_coords[0, 0, :] = torch.tensor([0.15, 0.25, 0.35])  # true source 0 near pred 0
target_coords[0, 1, :] = torch.tensor([-0.05, 0.45, 0.25]) # true source 1 near pred 1

valid_mask = torch.zeros(1, MAX_SOURCES, dtype=torch.bool)
valid_mask[0, 0] = True
valid_mask[0, 1] = True

matched_indices, count_targets = compute_minimum_cost_matching(pred_coords, target_coords, valid_mask)

print(f"    Matched prediction slots: {matched_indices[0]}")
print(f"    Derived count target: {count_targets[0].item()}")
assert len(matched_indices[0]) == 2, "Should match 2 predictions"
assert 0 in matched_indices[0], "Slot 0 should be matched"
assert 1 in matched_indices[0], "Slot 1 should be matched"
assert count_targets[0].item() == 2, "Count should be 2"

conf_labels = build_confidence_labels_from_matching(matched_indices, batch_size)
print(f"    Confidence labels: {conf_labels[0].tolist()}")
assert conf_labels[0, 0] == 1.0, "Slot 0 should have conf=1"
assert conf_labels[0, 1] == 1.0, "Slot 1 should have conf=1"
assert conf_labels[0, 2] == 0.0, "Slot 2 should have conf=0"
assert conf_labels[0, 3] == 0.0, "Slot 3 should have conf=0"
assert conf_labels[0, 4] == 0.0, "Slot 4 should have conf=0"

print("    ✓ Test 1 PASSED")

# Test case 2: Matching assigns non-sequential slots
print("\n  Test 2: Non-sequential slot matching")
pred_coords2 = torch.zeros(1, MAX_SOURCES, 3)
pred_coords2[0, 0, :] = torch.tensor([0.9, 0.9, 0.9])  # far away
pred_coords2[0, 1, :] = torch.tensor([-0.9, -0.9, 0.8]) # far away
pred_coords2[0, 2, :] = torch.tensor([0.1, 0.2, 0.3])  # CLOSE to true source
pred_coords2[0, 3, :] = torch.tensor([-0.1, 0.5, 0.2]) # CLOSE to true source
pred_coords2[0, 4, :] = torch.tensor([0.0, 0.0, 0.0])  # center

target_coords2 = torch.full((1, MAX_SOURCES, 3), float('nan'))
target_coords2[0, 0, :] = torch.tensor([0.15, 0.25, 0.35])  # should match slot 2
target_coords2[0, 1, :] = torch.tensor([-0.05, 0.45, 0.25]) # should match slot 3

matched_indices2, count_targets2 = compute_minimum_cost_matching(
    pred_coords2, target_coords2, valid_mask
)

print(f"    Matched prediction slots: {matched_indices2[0]}")
print(f"    Derived count target: {count_targets2[0].item()}")
assert len(matched_indices2[0]) == 2, "Should match 2 predictions"
assert 2 in matched_indices2[0], "Slot 2 should be matched (not slot 0!)"
assert 3 in matched_indices2[0], "Slot 3 should be matched (not slot 1!)"

conf_labels2 = build_confidence_labels_from_matching(matched_indices2, batch_size)
print(f"    Confidence labels: {conf_labels2[0].tolist()}")
assert conf_labels2[0, 0] == 0.0, "Slot 0 should have conf=0 (was far)"
assert conf_labels2[0, 1] == 0.0, "Slot 1 should have conf=0 (was far)"
assert conf_labels2[0, 2] == 1.0, "Slot 2 should have conf=1 (matched!)"
assert conf_labels2[0, 3] == 1.0, "Slot 3 should have conf=1 (matched!)"
assert conf_labels2[0, 4] == 0.0, "Slot 4 should have conf=0"

print("    ✓ Test 2 PASSED - correctly handles non-sequential matching!")

# Test case 3: Zero sources
print("\n  Test 3: Scene with zero sources")
empty_valid = torch.zeros(1, MAX_SOURCES, dtype=torch.bool)
empty_target = torch.full((1, MAX_SOURCES, 3), float('nan'))

matched_empty, count_empty = compute_minimum_cost_matching(
    pred_coords, empty_target, empty_valid
)
print(f"    Matched prediction slots: {matched_empty[0]}")
print(f"    Derived count target: {count_empty[0].item()}")
assert len(matched_empty[0]) == 0, "Should match nothing"
assert count_empty[0].item() == 0, "Count should be 0"

conf_empty = build_confidence_labels_from_matching(matched_empty, batch_size)
print(f"    Confidence labels: {conf_empty[0].tolist()}")
assert (conf_empty[0] == 0.0).all(), "All confidences should be 0"

print("    ✓ Test 3 PASSED")

print("\n" + "=" * 60)
print("  ✓ All matching-based confidence tests PASSED!")
print("=" * 60)
print("\n  Key fixes verified:")
print("    1. Confidence labels built AFTER matching (not pre-assigned)")
print("    2. Count targets derived from matching (not dataset)")
print("    3. Non-sequential slot matching handled correctly")
print("    4. Zero-source scenes handled correctly")
