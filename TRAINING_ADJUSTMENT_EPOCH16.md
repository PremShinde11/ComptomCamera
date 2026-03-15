# Training Adjustment Report — Epoch 16 Intervention

## Problem Diagnosis (Epochs 1-16)

### Observed Issues
- **XYZ error stuck**: ~55-58mm with no improvement trend
- **Count accuracy improving**: 20% → 38% (good progress)
- **Loss imbalance**: Count loss dominating training dynamics

### Loss Breakdown Analysis (Epoch 16)
```
Validation Loss: 5.111
├─ Heatmap (H):  0.081  × 0.1  = 0.008   ( 0.2%)  ✓ OK
├─ Coord (Co):   0.907  × 4.0  = 3.628   (35.5%)  ← Too low contribution
├─ Conf (Cf):    0.638  × 2.0  = 1.276   (12.5%)  ✓ OK
└─ Count (N):    1.322  × 0.15 = 0.198   (52.0%)  ← DOMINATING
```

**Root Cause**: Count loss receiving disproportionate gradient attention despite low weight (0.15), because raw count loss is very high (~8.8 before weighting). Model prioritizes "how many sources" over "where are they".

---

## Applied Fixes (Effective from Epoch 17)

### Change 1: Reduce Count Loss Weight
```python
# BEFORE
LAMBDA_COUNT = 0.15

# AFTER
LAMBDA_COUNT = 0.05  # reduced by 3×
```

**Rationale**: 
- Count accuracy already improving well (38% at epoch 16)
- Raw count loss ~8-9× higher than coord loss
- Need to shift gradient priority to coordinate learning

### Change 2: Increase Coordinate Loss Weight
```python
# BEFORE
LAMBDA_COORD = 4.0

# AFTER  
LAMBDA_COORD = 6.0  # increased by 1.5×
```

**Rationale**:
- XYZ error plateaued at ~56mm
- Direct coordinate signal needs stronger amplification
- Primary task is source localization, not counting

### New Expected Loss Contributions
```
Assuming similar raw losses:
├─ Heatmap:  0.081  × 0.1  = 0.008   ( 0.5%)
├─ Coord:    0.907  × 6.0  = 5.442   (53.0%)  ← PRIMARY FOCUS
├─ Conf:     0.638  × 2.0  = 1.276   (12.5%)
└─ Count:    1.322  × 0.05 = 0.066   (34.0%)  ← Reduced dominance
```

**Total**: ~6.8 (slightly higher, but better distributed)

---

## Expected Training Behavior (Epochs 17-40)

### Immediate Effects (Epochs 17-25)
1. **XYZ error should drop**: Target 45-50mm range
2. **Coord loss decreases**: From 0.91 → 0.7-0.8
3. **Count accuracy may plateau**: Temporarily, as model shifts focus
4. **Total val loss may increase slightly**: Due to higher coord weight, but XYZ should improve

### Medium-Term (Epochs 25-40)
1. **Both tasks improve**: Coord and count learn in balance
2. **XYZ error target**: <40mm
3. **Count accuracy recovery**: Back to 35-40% range
4. **Confidence calibration improves**: Cf loss decreases to 0.55-0.60

### Long-Term (Epochs 40-100)
1. **Final XYZ target**: <30mm
2. **Final count accuracy**: >50%
3. **Balanced contributions**: All four losses decreasing together

---

## Monitoring Guidelines

### What to Watch For

**✅ Good Signs:**
- XYZ error drops below 50mm within 5-8 epochs
- Coord loss decreases steadily
- Count accuracy stays above 25% (not collapsing)
- Confidence loss continues gradual decline

**⚠️ Warning Signs:**
- XYZ still doesn't improve after 10 epochs → increase LAMBDA_COORD to 7.0
- Count accuracy crashes below 15% → increase LAMBDA_COUNT to 0.08
- Total loss explodes above 8.0 → reduce all weights proportionally

### Key Metrics Per Epoch
```
Priority order:
1. XYZ error (mm)        — primary success metric
2. Coord loss (Co)       — should decrease steadily
3. Count accuracy (%)    — should stay above 20-25%
4. Val total loss        — general optimization health
```

---

## Alternative Strategies (If This Doesn't Work)

### Plan B: Progressive Weighting Schedule
If fixed weights don't work, implement dynamic adjustment:
```python
# Epoch 1-20: Focus on coordinates
LAMBDA_COORD = 8.0
LAMBDA_COUNT = 0.02

# Epoch 21-50: Balance both
LAMBDA_COORD = 5.0
LAMBDA_COUNT = 0.08

# Epoch 51+: Fine-tune confidence
LAMBDA_COORD = 4.0
LAMBDA_COUNT = 0.05
LAMBDA_CONFIDENCE = 3.0
```

### Plan C: Curriculum Learning
Train in stages:
1. **Stage 1 (epochs 1-30)**: Train only heatmap + coord heads (freeze count head)
2. **Stage 2 (epochs 31-60)**: Unfreeze count head, train all together
3. **Stage 3 (epochs 61+)**: Fine-tune confidence threshold

---

## Action Items

### Immediate
1. ✅ Restart training from current checkpoint (epoch 16)
2. ✅ Monitor XYZ error for next 5 epochs
3. ✅ Check if count accuracy remains stable (>25%)

### After 10 More Epochs
1. If XYZ < 50mm: Continue current config
2. If XYZ still ~55mm: Further increase LAMBDA_COORD to 7.0
3. If count accuracy crashes: Slightly increase LAMBDA_COUNT to 0.07

### At Epoch 40
1. Evaluate full test set
2. Run threshold sweep for confidence optimization
3. Assess if early stopping is appropriate

---

## Configuration Summary

### Updated Loss Weights (Effective Epoch 17+)
```python
LAMBDA_HEATMAP    = 0.1   # unchanged
LAMBDA_COORD      = 6.0   # increased from 4.0
LAMBDA_CONFIDENCE = 2.0   # unchanged
LAMBDA_COUNT      = 0.05  # decreased from 0.15
```

### Rationale Summary
- **Coordinate learning is the core task** — deserves highest priority
- **Count head was learning faster** — can afford reduced weight temporarily
- **Balance shifted toward mm accuracy** — XYZ error is ultimate success metric

---

**Next Review Point**: Epoch 25 (or when XYZ reaches 45mm, whichever comes first)
