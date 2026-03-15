# Verify Hungarian algorithm ensures one-to-one matching
import torch
from train import compute_minimum_cost_matching

print("=" * 60)
print("  Testing One-to-One Matching (Hungarian Algorithm)")
print("=" * 60)

# Test case: 2 true sources, 5 prediction slots
# Both true sources should match to DIFFERENT prediction slots
batch_size = 1
pred_coords = torch.zeros(1, 5, 3)
pred_coords[0, 0, :] = torch.tensor([10.0, 20.0, 5.0])   # CLOSE to true source 0
pred_coords[0, 1, :] = torch.tensor([10.5, 20.5, 5.5])  # ALSO CLOSE to true source 0
pred_coords[0, 2, :] = torch.tensor([-30.0, 40.0, -10.0]) # CLOSE to true source 1
pred_coords[0, 3, :] = torch.tensor([0.0, 0.0, 0.0])
pred_coords[0, 4, :] = torch.tensor([0.0, 0.0, 0.0])

target_coords = torch.full((1, 5, 3), float('nan'))
target_coords[0, 0, :] = torch.tensor([10.0, 20.0, 5.0])  # true source 0
target_coords[0, 1, :] = torch.tensor([-30.0, 40.0, -10.0]) # true source 1

valid_mask = torch.zeros(1, 5, dtype=torch.bool)
valid_mask[0, 0] = True
valid_mask[0, 1] = True

print("\n  Setup:")
print(f"    True source 0 at (10, 20, 5)")
print(f"    True source 1 at (-30, 40, -10)")
print(f"    Pred slot 0 at (10, 20, 5)   ← close to true 0")
print(f"    Pred slot 1 at (10.5, 20.5, 5.5) ← also close to true 0")
print(f"    Pred slot 2 at (-30, 40, -10) ← close to true 1")

matched_indices, count_targets = compute_minimum_cost_matching(
    pred_coords, target_coords, valid_mask
)

print(f"\n  Result:")
print(f"    Matched prediction slots: {matched_indices[0]}")
print(f"    Derived count target: {count_targets[0].item()}")

# CRITICAL TEST: No two true sources should claim the same prediction slot
pred_slots_list = list(matched_indices[0])
all_unique = len(pred_slots_list) == len(set(pred_slots_list))

print(f"\n  Verification:")
print(f"    Number of matched slots: {len(pred_slots_list)}")
print(f"    All slots unique: {all_unique}")
print(f"    Expected: True (one-to-one matching)")

if all_unique:
    print(f"\n  ✓ PASSED - Hungarian algorithm ensures one-to-one assignment!")
else:
    print(f"\n  ✗ FAILED - Multiple true sources claimed same slot!")
    print(f"     This would cause gradient conflict during training.")

print("\n" + "=" * 60)
