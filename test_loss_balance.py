import torch
import config
from model import ComptonSourceLocaliser
import torch.nn.functional as F
import numpy as np

print("="*65)
print("LOSS BALANCE SANITY CHECK")
print("="*65)
print()

# First verify count head init
model = ComptonSourceLocaliser()
final_layer = list(model.count_head.mlp.children())[-1]
print(f'Count head final weight std: {final_layer.weight.std().item():.4f}')
print(f'Expected: ~0.01  Status: {"OK" if final_layer.weight.std().item() < 0.05 else "INIT NOT APPLIED"}')
print()

losses = []
for trial in range(5):
    model = ComptonSourceLocaliser()
    events   = torch.randn(4, config.MAX_EVENTS_PER_SCENE, config.N_INPUT_FEATURES)
    pad_mask = torch.zeros(4, config.MAX_EVENTS_PER_SCENE, dtype=torch.bool)
    heatmap, coords_conf, count_logits = model(events, pad_mask)

    # Realistic targets
    t_heat  = torch.zeros(4, 1, config.HEATMAP_SIZE, config.HEATMAP_SIZE)
    t_coord = torch.rand(4, 5, 3) * 2 - 1
    t_conf  = torch.where(torch.rand(4,5) > 0.5,
                          torch.full((4,5), 0.9),
                          torch.full((4,5), 0.1))
    t_count = torch.randint(1, 6, (4,))

    coords   = coords_conf[:, :, :3]
    conf_raw = coords_conf[:, :, 3]

    lh  = F.binary_cross_entropy(heatmap, t_heat)
    lco = F.mse_loss(coords, t_coord)
    lcf = F.binary_cross_entropy(torch.sigmoid(conf_raw), t_conf)
    lct = F.cross_entropy(count_logits, t_count)

    wh  = lh.item()  * config.LAMBDA_HEATMAP
    wco = lco.item() * config.LAMBDA_COORD
    wcf = lcf.item() * config.LAMBDA_CONFIDENCE
    wct = lct.item() * config.LAMBDA_COUNT

    total = wh + wco + wcf + wct
    losses.append(total)
    pct = lambda x: f'{100*x/total:.0f}%'
    print(f'Trial {trial+1}: H={wh:.3f}({pct(wh)}) '
          f'Co={wco:.3f}({pct(wco)}) '
          f'Cf={wcf:.3f}({pct(wcf)}) '
          f'N={wct:.3f}({pct(wct)}) '
          f'Total={total:.3f}')

print()
print(f'Mean: {np.mean(losses):.3f}  Std: {np.std(losses):.3f}')
print(f'Target: mean 4-8, std < 0.3')
verdict = 'GOOD' if 4 <= np.mean(losses) <= 8 and np.std(losses) < 0.3 else 'ADJUST'
print(f'Verdict: {verdict}')
print()

if verdict == 'ADJUST':
    print("Current loss weights:")
    print(f"  LAMBDA_HEATMAP    = {config.LAMBDA_HEATMAP}")
    print(f"  LAMBDA_COORD      = {config.LAMBDA_COORD}")
    print(f"  LAMBDA_CONFIDENCE = {config.LAMBDA_CONFIDENCE}")
    print(f"  LAMBDA_COUNT      = {config.LAMBDA_COUNT}")
    print()
    print("Recommended adjustment:")
    print("  LAMBDA_COORD      = 4.0  (to make coord ~47% of total)")
    print("  LAMBDA_CONFIDENCE = 2.0  (to make conf ~22% of total)")
    print("  LAMBDA_COUNT      = 0.15 (already good after init fix)")
    print("  LAMBDA_HEATMAP    = 0.1  (keep)")
