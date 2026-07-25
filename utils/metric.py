import numpy as np

# None if target is empty
def calculate_dice(pred, target):
    """
    pred: numpy array
    target: numpy array
    """
    if np.sum(target) == 0:
        fp = np.sum(pred)
        return None
    else:
        intersection = np.sum(pred * target) * 2.0 + 1e-6
        union = np.sum(pred) + np.sum(target) + 1e-6
        return intersection / union

def calculate_sensitivity(pred, target):
    """
    pred: numpy array
    target: numpy array
    """
    if np.sum(target) == 0:
        return None
    else:
        return np.sum(pred * target) / np.sum(target)
    
def calculate_segmentation_metrics(predictions, masks):
    tp, fp, fn = [], [], []
    for pred, mask in zip(predictions, masks):
        temp_pred = pred.detach().cpu().numpy().squeeze()
        temp_mask = mask.detach().cpu().numpy().squeeze()

        if temp_pred.sum() == 0:
            fp.append(temp_mask.sum())
        else:
            tp.append((temp_pred * temp_mask).sum())
            fp.append(((1 - temp_mask) * temp_pred).sum())
            fn.append((temp_mask * (1 - temp_pred)).sum())
    return np.sum(tp), np.sum(fp), np.sum(fn)


def calculate_classification_metrics(cls_outputs, cls_targets):
    tp_cls = ((cls_outputs == 1) & (cls_targets == 1)).sum().item()
    tn_cls = ((cls_outputs == 0) & (cls_targets == 0)).sum().item()
    fp_cls = ((cls_outputs == 1) & (cls_targets == 0)).sum().item()
    fn_cls = ((cls_outputs == 0) & (cls_targets == 1)).sum().item()
    return tp_cls, tn_cls, fp_cls, fn_cls

