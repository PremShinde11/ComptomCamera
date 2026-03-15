# Training Quick Reference - Post-Fixes

## What Changed?

### Before (Broken)
```python
# Confidence labels pre-assigned to slots 0..N-1
conf_labels = dataset.confidence_labels  # WRONG!

# Count from dataset, confidence from matching → contradiction
count_target = dataset.count  # Could conflict with confidence BCE
```

### After (Fixed)
```python
# 1. Compute matching FIRST
matched_indices, count_from_matching = compute_minimum_cost_matching(...)

# 2. Build confidence FROM matching
conf_labels = build_confidence_labels_from_matching(matched_indices)

# 3. Use derived count (not dataset)
count_target = count_from_matching  # Perfectly aligned with confidence
```

---

## Start Training

```bash
python train.py --epochs 100 --batch-size 16
```

### Expected Output (per epoch)
```
  Epoch   45/100  | Train: 2.34 (H:0.23 C:0.51)  | Val: 2.45 (Acc:78.5%)  | XYZ:  28.42 mm  | LR: 1.00e-03  |  45.2s
                    ↑                              ↑                        ↑
                              Count accuracy (%)   Coordinate error (mm)
```

**Key metrics to watch:**
- `Acc:XX%` - Count accuracy (should increase over epochs, target >75%)
- `XYZ: XX.XX mm` - Spatial accuracy (should decrease, target <30mm)
- Both should improve together - if count accuracy stays low, coordinates will be misassigned

---

## Monitoring During Training

### Check training log
```bash
tail -f models/training_log.csv
```

### Key columns
```csv
epoch,val_count_acc,val_xyz_mm,val_conf_loss,val_count_loss
```

### Good signs ✓
- `val_count_acc` increases: 20% → 50% → 75%+
- `val_conf_loss` decreases: 0.9 → 0.7 → 0.5
- `val_xyz_mm` decreases steadily
- All 4 losses decrease together

### Warning signs ⚠️
- Count accuracy stuck <40% after epoch 30 → may need higher LAMBDA_COUNT
- Confidence loss not decreasing → check matching is working
- XYZ error plateaus early → may need higher LAMBDA_COORD

---

## Inference Behavior

### Example scene (3 true sources)

**Model outputs:**
```
Count logits: [-1.8, -2.1, 3.5, -0.9, -1.5, -2.8]  → pred_n = 2
Confidences:  [0.12,  0.08, 0.91,  0.34,  0.21,  0.05]
Coordinates:  [slot 0..5 positions]
```

**Selection process:**
1. Count predicts N=2 (argmax of logits)
2. Threshold filter: conf ≥ 0.3 → slots {2, 3} (n_above=2)
3. Final N = min(2, 2) = 2
4. Top-2 by confidence: slots {2, 3}
5. Return coordinates for slots 2 and 3

**Result:**
```
Prediction 1: (x,y,z) with conf=0.91
Prediction 2: (x,y,z) with conf=0.34
```

---

## Evaluation Output

### Run evaluation
```bash
python evaluate.py
```

### New CSV columns
```csv
pred_confidence,n_sources_pred
0.91,2
0.34,2
```

### Analysis possibilities
- Plot confidence histogram
- Accuracy-vs-confidence curve
- Optimize threshold post-training
- Compare predicted vs true count distribution

---

## Visual Inspection

### Run inspection
```bash
python inspect_scene.py --csv data/test/SIM_events_source_0000.csv
```

### Enhanced display

**Console:**
```
============================================================
  SIM_events_source_0000.csv
  True sources: 5  |  Predicted: 4  |  Type-1 events: 650
  Confidences: [0.91, 0.87, 0.74, 0.62]
  Src 0: True(-78.2,-39.6,-14.9) → Pred(-57.9,-34.7,2.5) (conf=0.91)  xyz=27.17mm
  ...
============================================================
```

**Metrics panel (in PNG):**
```
── COUNT PREDICTION ──
  True N        5
  Predicted N   4       ← red if wrong
  Confidences   [0.91, 0.87, 0.74, 0.62]

── Source 0 ──
  true    (-78.2, -39.6, -14.9)
  pred (conf=0.91)  (-57.9, -34.7, 2.5)
  XYZ err 27.17 mm
```

---

## Troubleshooting

### Issue: Count accuracy stays ~17% (random)
**Diagnosis:** Model not learning to count
**Solutions:**
1. Increase LAMBDA_COUNT: 0.5 → 1.0
2. Train longer (count often learns slower than coordinates)
3. Check that matching function returns correct counts

### Issue: All confidences ~0.5
**Diagnosis:** Confidence labels might be wrong
**Solutions:**
1. Verify matching is working (run test_matching.py)
2. Check that conf_labels are 0/1 (not all 1s)
3. Ensure derived count matches matched slot count

### Issue: Many false positives at low confidence
**Diagnosis:** Threshold too low or regularization missing
**Solutions:**
1. Increase CONFIDENCE_THRESHOLD: 0.3 → 0.4
2. Verify unmatched slots cluster at origin (check regularization)
3. Increase regularization weight: 0.1 → 0.2

### Issue: Unmatched slots drift far from origin
**Diagnosis:** Regularization not strong enough
**Solutions:**
1. Increase regularization weight: 0.1 → 0.3
2. Check matched_slot_mask is being passed correctly
3. Verify unmatched predictions actually receive gradient

---

## Hyperparameters Reference

### Current settings (config.py)
```python
MAX_SOURCES          = 5
CONFIDENCE_THRESHOLD = 0.3
LAMBDA_COUNT         = 0.5   # reduced from 2.0
LAMBDA_CONFIDENCE    = 1.0
LAMBDA_COORD         = 5.0
LAMBDA_HEATMAP       = 0.3
REGULARIZATION_WEIGHT = 0.1  # in model.py
```

### Tuning guidelines
- **Increase LAMBDA_COUNT** if count accuracy <50% after epoch 30
- **Increase LAMBDA_CONFIDENCE** if confidence doesn't correlate with accuracy
- **Increase REGULARIZATION_WEIGHT** if unmatched slots drift far from origin
- **Decrease CONFIDENCE_THRESHOLD** if missing real sources (low recall)
- **Increase CONFIDENCE_THRESHOLD** if many false positives (low precision)

---

## Success Criteria

After 100 epochs, expect:

| Metric | Target | How to check |
|--------|--------|--------------|
| Count accuracy | >75% | `val_count_acc` in log |
| Match rate | >90% | Eval summary |
| Mean XYZ error | <30mm | `val_xyz_mm` in log |
| Confidence-AUC | >0.85 | Plot from CSV |
| Unmatched slots near origin | <0.2 norm | Inspect predictions |

If all criteria met → model ready for physics analysis!

---

## Post-Training Analysis

### Confidence calibration plot
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/evaluation_results.csv')
matched = df[df['matched'] == True]

plt.hist(matched['pred_confidence'], bins=20, alpha=0.7)
plt.xlabel('Predicted Confidence')
plt.ylabel('Count')
plt.title('Confidence Distribution')
plt.show()
```

### Accuracy-vs-confidence
```python
from sklearn.metrics import roc_curve

# Binary label: correct if xyz error < 50mm
matched['correct'] = (matched['err_xyz'] < 50).astype(int)

fpr, tpr, _ = roc_curve(matched['correct'], matched['pred_confidence'])
auc = np.trapz(tpr, fpr)

plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Confidence Calibration')
plt.legend()
plt.show()
```

### Optimize threshold
```python
thresholds = np.arange(0.1, 0.9, 0.05)
for thresh in thresholds:
    above = matched[matched['pred_confidence'] >= thresh]
    precision = (above['err_xyz'] < 50).mean()
    recall = len(above) / len(matched)
    print(f"Threshold {thresh:.2f}: Precision={precision:.1%}, Recall={recall:.1%}")
```

Choose threshold based on your precision/recall trade-off needs.

---

## Files Modified Checklist

Before training, verify these files have the fixes:

- [ ] `config.py` - LAMBDA_COUNT = 0.5
- [ ] `model.py` - Regularization term in loss
- [ ] `train.py` - Matching functions, derived labels
- [ ] `evaluate.py` - Confidence logging
- [ ] `inspect_scene.py` - Count display

Run tests to confirm:
```bash
python test_new_arch.py && python test_matching.py
```

Both should pass ✓

---

## Ready to Train!

All critical bugs fixed. Architecture is sound. Tests pass.

```bash
python train.py --epochs 100 --batch-size 16
```

Monitor count accuracy - it should steadily increase from ~20% to >75%. If it does, everything else will follow.

Good luck! 🎯
