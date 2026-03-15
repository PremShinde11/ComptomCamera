# Quick Reference: Training Adjustment at Epoch 16

## What Changed

```python
# config.py - Updated values
LAMBDA_COORD = 6.0   # was 4.0 (+50%)
LAMBDA_COUNT = 0.05  # was 0.15 (-67%)
```

## Why We Did This

**Problem**: Count loss dominating (52% of weighted loss), XYZ error stuck at ~56mm  
**Solution**: Shift gradient priority from counting to localization

## Expected Results

### Next 5 Epochs (17-22)
- XYZ error: 56mm → 45-50mm ✓
- Coord loss: 0.91 → 0.75-0.85 ✓
- Count accuracy: May dip slightly (OK if >25%) ⚠️
- Total loss: May increase slightly (expected due to higher coord weight)

### Next 15 Epochs (17-32)
- XYZ error target: <40mm
- Count accuracy recovery: 35-40%
- Both losses decreasing together

## What to Monitor

**Check every epoch:**
1. XYZ error — should trend down within 3-5 epochs
2. Coord loss (Co) — primary learning signal now
3. Count accuracy — should stay above 25%

**Red flags (need intervention):**
- XYZ still >55mm after epoch 22 → Increase LAMBDA_COORD to 7.0
- Count accuracy <20% for 3+ epochs → Increase LAMBDA_COUNT to 0.07
- Total loss >8.0 → Reduce all weights by 20%

## Commands

### Continue Training (from epoch 16 checkpoint)
```bash
python train.py --epochs 100 --batch-size 16
```

Training will automatically resume from `models/best_model.pth` (epoch 16).

### Check Progress After 5 Epochs
```bash
# Look at last 5 lines of training log
tail -n 5 models/training_log.csv
```

Expected at epoch 22:
- XYZ: 45-50mm
- Co: 0.75-0.85
- Acc: 25-35%

### Emergency Stop (if something breaks)
If count accuracy crashes or loss explodes:
1. Press Ctrl+C to stop training
2. Revert config changes: LAMBDA_COORD=4.0, LAMBDA_COUNT=0.15
3. Resume from epoch 16 checkpoint

## Success Criteria at Epoch 30

✅ **Good scenario** (continue current config):
- XYZ < 40mm
- Count accuracy > 30%
- All losses decreasing

⚠️ **Mixed scenario** (minor tweak needed):
- XYZ 40-50mm but improving → continue 5 more epochs
- Count accuracy 20-30% → increase LAMBDA_COUNT to 0.07

❌ **Bad scenario** (major change needed):
- XYZ still >50mm → implement Plan B (progressive weighting)
- Count accuracy <15% → revert to original weights

## Files Modified

1. `config.py` — loss weight adjustments
2. `TRAINING_ADJUSTMENT_EPOCH16.md` — full analysis
3. `QUICK_REFERENCE.md` — this file

## Contact Points

**Decision tree:**
```
Epoch 22 check:
├─ XYZ < 50mm? 
│  ├─ YES → Continue to epoch 30
│  └─ NO → Increase LAMBDA_COORD to 7.0
│
├─ Count Acc > 25%?
│  ├─ YES → Perfect, continue
│  └─ NO → Wait 3 epochs, then decide
│
└─ Both improving?
   ├─ YES → On track!
   └─ NO → Consider Plan C (curriculum learning)
```

---

**Bottom line**: We shifted priority from counting to localization. XYZ should drop within 5 epochs. If it doesn't, we'll try Plan B.
