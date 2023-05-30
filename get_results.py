import glob
import pandas as pd

from src.utils.vis_utils import get_df, average_df

LOG_NAME = "logs.txt"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

root = "output"
df_list=[]
for seed in ["42", "44", "82", "100", "800"]:
#     model_type = f"adapter_{r}"
    files = glob.glob(f"{root}/seed{seed}/*/sup_vitb16_imagenet21k/*/*/{LOG_NAME}")
    for f in files:
        df = get_df(files, f"seed{seed}", root, is_best=False, is_last=True)
        if df is None:
            continue
        df["seed"] = seed
    df_list.append(df)

df= pd.concat(df_list)
df["type"] = "VPT"
print(df)
df.to_csv("output/VPT.csv")

# LR represents the base learning rate, not the scaled one.
f_df = average_df(df, metric_names=["l-test_top1"], take_average=True)
print(f_df)
f_df.to_csv("output/VPT_average.csv")

# df_list=[]
# for seed in ["42", "44", "82", "100", "800"]:
# #     model_type = f"adapter_{r}"
#     files = glob.glob(f"{root}/seed_new{seed}/*/sup_vitb16_imagenet21k/*/*/{LOG_NAME}")
#     for f in files:
#         df = get_df(files, f"seed_new{seed}", root, is_best=False, is_last=True)
#         if df is None:
#             continue
#         df["seed"] = seed
#     df_list.append(df)
#
# df= pd.concat(df_list)
# df["type"] = "CoVPT"
# print(df)
# df.to_csv("output/CoVPT.csv")
#
# # LR represents the base learning rate, not the scaled one.
# f_df = average_df(df, metric_names=["l-test_top1"], take_average=True)
# print(f_df)
# f_df.to_csv("output/CoVPT_average.csv")
