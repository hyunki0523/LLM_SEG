
from functools import lru_cache
import itertools
from typing import Union, Tuple, List, Optional, Any, Dict

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()


@lru_cache(maxsize=2)
def compute_gaussian(
    tile_size: Union[Tuple[int, ...], List[int]],
    sigma_scale: float = 1. / 8,
    value_scaling_factor: float = 1,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device('cuda', 0),
) -> torch.Tensor:
    tmp = np.zeros(tile_size, dtype=np.float32)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * float(value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    # cannot be 0 (avoid nans)
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map


def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...],
    tile_size: Tuple[int, ...],
    tile_step_size: float,
) -> List[List[int]]:
    assert all(i >= j for i, j in zip(image_size, tile_size)), "image size must be >= patch_size"
    assert 0 < tile_step_size <= 1, "step_size must be (0, 1]"

    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]
    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)
    return steps


def _internal_get_sliding_window_slicers(
    image_size: Tuple[int, ...],
    patch_size: Tuple[int, ...],
    tile_step_size: float = 0.25,
):
    slicers = []
    if len(patch_size) < len(image_size):
        # 2D patch over 3D volume (treat first spatial dim as "slices")
        steps = compute_steps_for_sliding_window(image_size[1:], patch_size, tile_step_size)
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), patch_size)]])
                    )
    else:
        steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]])
                    )
    return slicers


def _unwrap_model_output(out: Any) -> torch.Tensor:
    """
    model output이 tensor / (tensor, ...) / dict 형태여도 logits tensor로 통일.
    - tensor: 그대로
    - tuple/list: 첫 원소
    - dict: seg/logits/out/pred 순으로 탐색
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        if torch.is_tensor(out[0]):
            return out[0]
        return _unwrap_model_output(out[0])
    if isinstance(out, dict):
        for k in [
            "fused_logits",
            "vision_logits",
            "seg",
            "logits",
            "out",
            "pred",
            "prediction",
        ]:
            if k in out:
                return _unwrap_model_output(out[k])
        # 마지막 fallback: 첫 value
        if len(out) > 0:
            return _unwrap_model_output(next(iter(out.values())))
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _batch_model_kwargs(
    model_kwargs: Optional[Dict[str, Any]],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Move case-level conditioning to device and repeat it for SW patches."""
    batched = {}
    for key, value in (model_kwargs or {}).items():
        if not torch.is_tensor(value):
            batched[key] = value
            continue
        value = value.to(device, non_blocking=False)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, *value.shape[1:])
        elif value.shape[0] != batch_size:
            raise ValueError(
                f"Model kwarg '{key}' has batch={value.shape[0]}, expected 1 or {batch_size}."
            )
        batched[key] = value
    return batched


def _forward_model(
    x: torch.Tensor,
    model,
    context=None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    kwargs = _batch_model_kwargs(model_kwargs, int(x.shape[0]), x.device)
    if context is not None:
        return _unwrap_model_output(model(x, context, **kwargs))
    return _unwrap_model_output(model(x, **kwargs))


def _internal_maybe_mirror_and_predict(
    x: torch.Tensor,
    model,
    mirror_axes: Optional[List[int]] = None,
    context=None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    # base prediction
    prediction = _forward_model(
        x, model, context=context, model_kwargs=model_kwargs
    )

    if mirror_axes is None:
        return prediction

    # x: 5d for 3D (B,C,D,H,W), 4d for 2D (B,C,H,W)
    assert max(mirror_axes) <= x.ndim - 3, "mirror_axes does not match input dimension"

    # axes are shifted by +2 (skip B,C)
    axes_combinations = [
        c for i in range(len(mirror_axes))
        for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
    ]
    axes_combinations = axes_combinations[:3]  # keep legacy behavior

    for axes in axes_combinations:
        x_flipped = torch.flip(x, axes)
        pred_flipped = _forward_model(
            x_flipped, model, context=context, model_kwargs=model_kwargs
        )
        prediction = prediction + torch.flip(pred_flipped, axes)

    prediction = prediction / (len(axes_combinations) + 1)
    return prediction


def _infer_out_channels_from_first_patch(
    model,
    data: torch.Tensor,
    slicers,
    device: torch.device,
    mirror_axes: Optional[List[int]] = None,
    context=None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[int, torch.Tensor]:
    """
    첫 번째 patch로 1회 forward 해서 out_ch를 추정.
    또한 첫 patch prediction을 반환해서 loop에서 재사용(중복 forward 방지).
    Returns: (out_ch, pred0) where pred0 shape is [out_ch, *patch_spatial]
    """
    first_sl = slicers[0]
    workon = data[first_sl][None].to(device, non_blocking=False)  # [1, C_in, ...]
    pred0 = _internal_maybe_mirror_and_predict(
        workon,
        model,
        mirror_axes=mirror_axes,
        context=context,
        model_kwargs=model_kwargs,
    )
    pred0 = _unwrap_model_output(pred0)
    if pred0.ndim < 3:
        raise RuntimeError(f"Unexpected prediction ndim={pred0.ndim}, shape={tuple(pred0.shape)}")
    if pred0.shape[0] != 1:
        # 대부분 B=1 유지 가정. 아니면 첫 배치만 사용.
        pred0 = pred0[:1]
    out_ch = int(pred0.shape[1])
    return out_ch, pred0[0]  # [out_ch, *patch_spatial]


def _internal_predict_sliding_window_return_logits(
    model,
    data: torch.Tensor,
    slicers,
    patch_size: Tuple[int, ...],
    do_on_device: bool = True,
    device: torch.device = torch.device('cpu'),
    use_gaussian: bool = True,
    mirror_axes: Optional[List[int]] = None,
    context=None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    accumulation_dtype: torch.dtype = torch.float16,
    sw_batch_size: int = 1,
):
    predicted_logits = n_predictions = gaussian = None
    results_device = device if do_on_device else torch.device('cpu')

    try:
        empty_cache(device)

        # move full data to results device (usually same as device)
        data = data.to(results_device)

        # 1) infer output channels dynamically (and reuse first patch prediction)
        out_ch, pred0 = _infer_out_channels_from_first_patch(
            model=model,
            data=data,
            slicers=slicers,
            device=device,
            mirror_axes=mirror_axes,
            context=context,
            model_kwargs=model_kwargs,
        )

        # 2) allocate buffers
        predicted_logits = torch.zeros(
            (out_ch, *data.shape[1:]),
            dtype=accumulation_dtype,
            device=results_device,
        )
        n_predictions = torch.zeros(
            data.shape[1:],
            dtype=accumulation_dtype,
            device=results_device,
        )

        # 3) gaussian importance map
        gaussian = None
        if use_gaussian:
            gaussian = compute_gaussian(
                tuple(patch_size),
                sigma_scale=1. / 8,
                value_scaling_factor=10,
                dtype=accumulation_dtype,
                device=results_device,
            )

        # 4) loop over patches in mini-batches of sw_batch_size
        n_slicers = len(slicers)
        pbar = tqdm(range(0, n_slicers, sw_batch_size), desc=f"SW (bs={sw_batch_size})")
        
        for batch_start in pbar:
            batch_end = min(batch_start + sw_batch_size, n_slicers)
            batch_slicers = slicers[batch_start:batch_end]
            
            # Collect patches into a batch tensor
            batch_patches = []
            for idx_in_batch, sl in enumerate(batch_slicers):
                global_idx = batch_start + idx_in_batch
                if global_idx == 0:
                    # 첫 번째 패치는 이미 계산됨 (pred0 재사용)
                    batch_patches.append(None)  # placeholder
                else:
                    patch = data[sl][None]  # [1, C_in, D', H', W']
                    batch_patches.append(patch)
            
            # Forward: 첫 번째 패치(pred0) 외의 패치들을 배치로 묶어서 한 번에 처리
            non_first_patches = [p for p in batch_patches if p is not None]
            pred_results = []
            
            if non_first_patches:
                batched_input = torch.cat(non_first_patches, dim=0).to(device, non_blocking=False)  # [B, C_in, ...]
                batched_pred = _internal_maybe_mirror_and_predict(
                    batched_input,
                    model,
                    mirror_axes=mirror_axes,
                    context=context,
                    model_kwargs=model_kwargs,
                )
                batched_pred = _unwrap_model_output(batched_pred)
                batched_pred = batched_pred.to(results_device, non_blocking=False)
            
            # Accumulate results
            non_first_idx = 0
            for idx_in_batch, sl in enumerate(batch_slicers):
                global_idx = batch_start + idx_in_batch
                if global_idx == 0:
                    pred_patch = pred0.to(results_device, non_blocking=False)
                else:
                    pred_patch = batched_pred[non_first_idx]  # [out_ch, *patch_spatial]
                    non_first_idx += 1
                
                if use_gaussian:
                    predicted_logits[sl] += (pred_patch * gaussian)
                    n_predictions[sl[1:]] += gaussian
                else:
                    predicted_logits[sl] += pred_patch
                    n_predictions[sl[1:]] += 1

        predicted_logits = predicted_logits / n_predictions

        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError(
                "Encountered inf in predicted logits. "
                "Reduce value_scaling_factor in compute_gaussian or use fp32 accumulation."
            )

    except Exception as e:
        del predicted_logits, n_predictions, gaussian
        empty_cache(device)
        empty_cache(results_device)
        raise e

    return predicted_logits


@torch.no_grad()
def predict_sliding_window_return_logits(
    model: torch.nn.Module,
    input_image: torch.Tensor,            # (C, D, H, W) or (C, H, W)
    patch_size: Tuple[int, ...],
    device: torch.device = torch.device('cpu'),
    step_size: float = 0.5,
    mirror_axes: Optional[List[int]] = None,
    use_gaussian: bool = True,
    context=None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    pad_value: float = -1.0,              # train과 동일하게 [-1,1] 입력이면 -1이 맞음
    accumulation_dtype: torch.dtype = torch.float16,
    sw_batch_size: int = 1,
) -> torch.Tensor:
    assert isinstance(input_image, torch.Tensor)

    # pad to at least patch_size
    data, slicer_revert_padding = pad_nd_image(
        input_image,
        patch_size,
        'constant',
        {'value': float(pad_value)},
        True,
        None
    )

    slicers = _internal_get_sliding_window_slicers(tuple(data.shape[1:]), patch_size, step_size)

    predicted_logits = _internal_predict_sliding_window_return_logits(
        model=model,
        data=data,
        slicers=slicers,
        patch_size=patch_size,
        do_on_device=True,
        device=device,
        mirror_axes=mirror_axes,
        use_gaussian=use_gaussian,
        context=context,
        model_kwargs=model_kwargs,
        accumulation_dtype=accumulation_dtype,
        sw_batch_size=sw_batch_size,
    )

    empty_cache(device)

    # revert padding
    predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
    return predicted_logits
