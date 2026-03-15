# Quick Start Guide - New Architecture

## Training the Updated Model

### Step 1: Verify Architecture
```bash
python test_new_arch.py
```
Expected: All tests pass ✓

### Step 2: Start Training
```bash
python train.py
```

Or with custom parameters:
```bash
python train.py --epochs 100 --batch-size 16
```

### What's Different Now?

**Old Training:**
- Single loss value logged
- Always predicted 5 sources
- No confidence scores

**New Training:**
- 4 loss components logged separately:
  - Heatmap loss (BCE)
  - Coordinate loss (MSE)
  - Confidence loss (BCE)
  - Count loss (CrossEntropy)
- Predicts variable source count 0-5
- Each prediction has confidence score

### Monitoring Training

Check `models/training_log.csv` during training:
```csv
epoch,train_loss,train_heat,train_coord,train_conf,train_count,val_loss,val_heat,val_coord,val_conf,val_count,val_xyz_mm,lr,time_s
```

Good signs:
- All 4 loss components decreasing
- val_xyz_mm improving
- count_loss converging (model learns to predict correct N)

### Expected Behavior

**Early epochs (1-20):**
- High count_loss (~2.0-3.0)
- Model learning to predict source count
- Coordinate loss may be unstable

**Mid training (20-50):**
- Count accuracy improves
- Confidence loss decreases
- Predictions become more reliable

**Late training (50+):**
- Count prediction stabilizes
- Confidence correlates with accuracy
- XYZ error continues decreasing

## Evaluation

### Test Set Evaluation
```bash
python evaluate.py
```

Output includes:
- Source count accuracy
- Matched vs missed sources
- Error breakdown by scene complexity

### Visual Inspection
```bash
python inspect_scene.py --csv data/test/SIM_events_source_0000.csv
```

Shows:
- True vs predicted source count in title
- Confidence-filtered predictions
- Matching between true and predicted sources

## Inference Details

### Prediction Pipeline

For each scene:
1. **Count Prediction**: `pred_N = argmax(count_logits)`
2. **Confidence Extraction**: `conf = sigmoid(coord[:, 3])`
3. **Ranking**: Sort by confidence descending
4. **Selection**: Keep top-pred_N predictions
5. **Filtering**: Remove conf < CONFIDENCE_THRESHOLD

### Example Output

Scene with 3 true sources:
```
Count logits: [−2.1, −1.5, 3.2, −0.8, −1.2, −2.5]  → pred_N = 2
Predictions (before filtering):
  Src 0: (x,y,z) conf=0.92  ← keep (top-2 by conf)
  Src 1: (x,y,z) conf=0.87  ← keep (top-2 by conf)
  Src 2: (x,y,z) conf=0.31  ← discard (not in top-2)
  Src 3: (x,y,z) conf=0.15  ← discard (not in top-2)
  Src 4: (x,y,z) conf=0.08  ← discard (not in top-2)

Final: 2 predictions returned
```

## Hyperparameters

### Loss Weights (config.py)
```python
LAMBDA_HEATMAP    = 0.3   # Spatial heatmap quality
LAMBDA_COORD      = 5.0   # XYZ coordinate accuracy
LAMBDA_CONFIDENCE = 1.0   # Confidence calibration
LAMBDA_COUNT      = 2.0   # Count classification accuracy
```

Tune if needed:
- Increase `LAMBDA_COUNT` if count accuracy is poor
- Increase `LAMBDA_CONFIDENCE` if confidence doesn't match accuracy
- Increase `LAMBDA_COORD` for better XYZ precision

### Confidence Threshold
```python
CONFIDENCE_THRESHOLD = 0.3
```

Lower (0.2): More predictions, higher recall, lower precision
Higher (0.5): Fewer predictions, higher precision, lower recall

## Troubleshooting

### Issue: Count always predicts 5
**Solution:** Increase LAMBDA_COUNT or train longer

### Issue: All confidences ~0.5
**Solution:** Check confidence_labels are 0/1 (not all 1s)

### Issue: Many false positives
**Solution:** Increase CONFIDENCE_THRESHOLD or LAMBDA_CONFIDENCE

### Issue: Missing real sources
**Solution:** Decrease CONFIDENCE_THRESHOLD or increase LAMBDA_COORD

## Performance Metrics

After training, expect:
- **Count accuracy**: >80% exact match
- **Match rate**: >90% of true sources matched
- **XYZ error**: <30mm mean (depends on dataset difficulty)

## Next Steps After Training

1. Run full test set evaluation
2. Inspect representative scenes
3. Compare metrics vs old architecture
4. Analyze failure cases (wrong count, low confidence)
5. Consider per-source heatmap decoding (advanced)
