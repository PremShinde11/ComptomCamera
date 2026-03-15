# Architecture Update Summary

## Changes Implemented

### 1. Configuration (config.py)
Added new hyperparameters for multi-source prediction:
- `MAX_SOURCES = 5` - Maximum number of sources per scene
- `CONFIDENCE_THRESHOLD = 0.3` - Discard predictions below this confidence
- `LAMBDA_COUNT = 2.0` - Weight for count classification loss
- `LAMBDA_CONFIDENCE = 1.0` - Weight for per-prediction confidence loss

### 2. Model Architecture (model.py)

#### New Component: CountHead
- Predicts source count (0-5) as classification task
- Output: raw logits for 6 classes (0,1,2,3,4,5)
- Architecture: MLP with ReLU activation

#### Modified Component: CoordinateMlpHead
- Changed output from `(MAX_SOURCES, 3)` to `(MAX_SOURCES, 4)`
- 4th dimension is raw confidence logit (apply sigmoid for probability)
- Each source prediction now includes confidence score

#### Updated Forward Pass
```python
heatmap, coords_and_conf, count_logits = model(events, padding_mask)
```
- `heatmap`: (batch, 1, 64, 64) - spatial heatmap
- `coords_and_conf`: (batch, MAX_SOURCES, 4) - (x,y,z,confidence_logit)
- `count_logits`: (batch, MAX_SOURCES+1) - count classification logits

#### Enhanced Loss Function
New 4-component loss:
```python
total_loss = (
    lambda_heatmap    * loss_heatmap +      # BCE on heatmap
    lambda_coord      * loss_coord +        # MSE on coordinates
    lambda_confidence * loss_confidence +   # BCE on confidence labels
    lambda_count      * loss_count          # CrossEntropy on count
)
```

### 3. Training Pipeline (train.py)

#### Dataset Updates
- Added `build_confidence_labels()` function
- Confidence label = 1 for real sources, 0 for padding/absent slots
- Returns count_target (true number of sources 0-5)

#### Loss Computation
- Uses updated `ComptonLocalisationLoss` from model.py
- Computes all 4 loss components
- Logs individual losses for monitoring

#### Training Metrics
Now tracks:
- Total loss
- Heatmap loss
- Coordinate loss  
- Confidence loss
- Count loss
- XYZ error (mm)

### 4. Inference Pipeline (evaluate.py, inspect_scene.py)

#### New Prediction Strategy
1. Get predicted count N from argmax(count_logits)
2. Extract confidence scores via sigmoid on 4th coordinate
3. Sort predictions by confidence descending
4. Keep top-N predictions where N = predicted count
5. Filter out predictions with confidence < threshold

#### Before vs After
**Before:**
- Always predicted exactly MAX_SOURCES=5
- No confidence filtering
- All predictions treated equally

**After:**
- Predicts variable count 0-5
- Confidence-based filtering
- More interpretable outputs

## Benefits

1. **Interpretability**: Model explicitly predicts how many sources exist
2. **Confidence Scoring**: Each prediction has associated confidence
3. **Flexible Output**: Can predict 0-5 sources, not always 5
4. **Better Training**: Separate loss terms for count and confidence give clearer gradients
5. **Reduced False Positives**: Confidence threshold filters weak predictions

## Next Steps

### Priority 1: Retraining
Train new model with updated architecture:
```bash
python train.py --epochs 100 --batch-size 16
```

Expected improvements:
- Better source count accuracy
- Lower false positive rate
- More reliable confidence estimates

### Priority 2: Per-Source Heatmap Decoding (Optional)
After count prediction is working well:
- Implement cluster-based heatmap decoding
- One peak per predicted source instead of summed blobs
- Requires changes to SpatialHeatmapDecoder

## Files Modified

1. `config.py` - Added new hyperparameters
2. `model.py` - CountHead, 4D coordinate output, enhanced loss
3. `train.py` - Confidence labels, count targets, updated training loop
4. `evaluate.py` - New inference with count+confidence filtering
5. `inspect_scene.py` - Updated visualization for new output format

## Testing

Run test script to verify architecture:
```bash
python test_new_arch.py
```

Expected output:
- Forward pass shapes correct
- All 4 loss components computed
- No errors

===================================================
=========                                            Testing New Architecture with Count + Confidence
===================================================
=========                                          
  Device: cpu
  MAX_SOURCES: 5
  LAMBDA_COUNT: 2.0
  LAMBDA_CONFIDENCE: 1.0
C:\Users\premy\AppData\Local\Programs\Python\Python
311\Lib\site-packages\torch\nn\modules\transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True                           warnings.warn(

  Model parameters: 957,435

  Running forward pass...
  Heatmap shape:      (2, 1, 64, 64)
  Coords+Conf shape:  (2, 5, 4)
  Count logits shape: (2, 6)

  ✓ All shapes correct!

  Computing loss...
  Total loss:     16.7841
  Heatmap loss:   0.8026
  Coord loss:     2.0277
  Conf loss:      0.8632
  Count loss:     2.7709

===================================================
=========                                            ✓ Architecture test PASSED!
===================================================