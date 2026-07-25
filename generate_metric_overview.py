import glob
import os
import sys

import numpy as np
import pandas as pd


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


if len(sys.argv) > 1:
    folder_path = sys.argv[1]
else:
    print("Usage: python generate_metric_overview.py <metric_result_folder>")
    sys.exit(1)


csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [
    f for f in csv_files
    if "overview" not in os.path.basename(f).lower()
    and "excluded_cases" not in os.path.basename(f).lower()
]

summary_list = []

for file in csv_files:
    file_name = os.path.basename(file)
    try:
        df = pd.read_csv(file)

        if df.empty or "dice" not in df.columns:
            print(f"Skipping invalid metric file: {file_name}")
            continue

        df["gt_has_lesion"] = to_bool(df["gt_has_lesion"])
        df["pred_has_lesion"] = to_bool(df["pred_has_lesion"])
        if "metric_include_hemo_dice" in df.columns:
            df["metric_include_hemo_dice"] = to_bool(df["metric_include_hemo_dice"])
        else:
            df["metric_include_hemo_dice"] = df["gt_has_lesion"]
        if "class_hemo" in df.columns:
            df["class_hemo"] = to_bool(df["class_hemo"])
        else:
            df["class_hemo"] = False

        for col in [
            "dice",
            "hd95_mm",
            "pixel_sensitivity",
            "gt_voxels",
            "pred_voxels",
            "tp_voxels",
            "fn_voxels",
            "fp_voxels",
        ]:
            if col in df.columns:
                df[col] = numeric(df[col])

        hemo_df = df[df["metric_include_hemo_dice"]].copy()
        mask_positive_df = df[df["gt_has_lesion"]].copy()
        excluded_df = df[
            (df["gt_has_lesion"])
            & (~df["metric_include_hemo_dice"])
        ].copy()

        mean_dice = hemo_df["dice"].mean() if not hemo_df.empty else np.nan
        mean_dice_mask_positive = (
            mask_positive_df["dice"].mean() if not mask_positive_df.empty else np.nan
        )
        mean_hd95 = hemo_df["hd95_mm"].mean() if not hemo_df.empty else np.nan
        mean_sens_hemo = (
            hemo_df["pixel_sensitivity"].mean() if not hemo_df.empty else np.nan
        )
        mean_sens_all = df["pixel_sensitivity"].mean()

        total_cases = len(df)
        fp_df = df[(df["gt_has_lesion"] == False) & (df["pred_has_lesion"] == True)]
        fn_df = hemo_df[
            (hemo_df["gt_has_lesion"] == True)
            & (hemo_df["pred_has_lesion"] == False)
        ]

        fp_ratio = len(fp_df) / total_cases if total_cases > 0 else 0
        mean_fp_voxels = fp_df["pred_voxels"].mean() if not fp_df.empty else 0
        fn_ratio = len(fn_df) / len(hemo_df) if len(hemo_df) > 0 else 0
        mean_fn_voxels = fn_df["fn_voxels"].mean() if not fn_df.empty else 0

        summary_list.append(
            {
                "File_Name": file_name,
                "Mean_Dice": round(mean_dice, 4) if pd.notna(mean_dice) else None,
                "Mean_Dice_MaskPositive": (
                    round(mean_dice_mask_positive, 4)
                    if pd.notna(mean_dice_mask_positive)
                    else None
                ),
                "Hemo_Dice_Cases": int(len(hemo_df)),
                "Mask_Positive_Cases": int(len(mask_positive_df)),
                "Excluded_MaskPositive_NonHemoClass": int(len(excluded_df)),
                "Mean_HD95_mm": round(mean_hd95, 2) if pd.notna(mean_hd95) else None,
                "Mean_Sensitivity": round(mean_sens_all, 4),
                "Mean_Sensitivity_HemoDiceSet": (
                    round(mean_sens_hemo, 4) if pd.notna(mean_sens_hemo) else None
                ),
                "FP_Ratio": round(fp_ratio, 4),
                "Mean_FP_Voxels": round(mean_fp_voxels, 2),
                "FN_Ratio": round(fn_ratio, 4),
                "Mean_FN_Voxels": round(mean_fn_voxels, 2),
            }
        )

    except Exception as e:
        print(f"Error while processing {file_name}: {e}")


overview_df = pd.DataFrame(summary_list)
if not overview_df.empty and "Mean_Dice" in overview_df.columns:
    overview_df = overview_df.sort_values("Mean_Dice", ascending=False)

output_path = os.path.join(folder_path, "overview_results.csv")
overview_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nOverview complete: {len(summary_list)} files")
print(f"Saved to: {output_path}")
