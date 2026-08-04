# v3.4 Partial Deep Supervision

## Objective

Measure the effect of decoder deep supervision independently of additional
training time. All three branches start from the same completed v3a
`no_soft_cached` final weight and use the same frozen-text cache.

## Experiment matrix

- [ ] `cached_control` on GPU 0,1: 60 epochs, no deep supervision
- [ ] `cached_ds2` on GPU 2,3: weights `1.0,0.3`
- [ ] `cached_ds3` on GPU 4,5: weights `1.0,0.3,0.1`

Common settings: batch 4/GPU, accumulation 4, global effective batch 32,
64 iterations/epoch, LR `3e-6`, fresh optimizer, checkpoint every 5 epochs.

## Acceptance gates

- [ ] All three smoke tests finish without OOM, DDP unused-parameter errors,
      missing cache prompts, or non-finite loss.
- [ ] Every branch loads the same v3a final weight and starts at epoch 1 with
      a fresh optimizer.
- [ ] Validation and inference use only the final full-resolution output.
- [ ] `cached_ds2` or `cached_ds3` improves validation Dice without materially
      increasing false-positive voxels or predicted-volume ratio.
- [ ] Report small-lesion Dice=0 rate and sensitivity in addition to mean Dice.

## Interpretation

- `cached_ds2 > cached_control`: high-resolution auxiliary supervision helps.
- `cached_ds3 > cached_ds2`: the additional middle scale is useful.
- `cached_ds3 < cached_ds2`: the coarser target is erasing or enlarging small
  hemorrhages; retain two supervised outputs.
- Both DS branches worse than control: decoder supervision is not the current
  bottleneck; prioritize grounding/gating instead.
