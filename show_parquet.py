import pandas as pd

# df = pd.read_parquet("data/lerobot/libero/data/chunk-000/episode_000000.parquet")
df = pd.read_parquet("data/lerobot/express_v3/data/chunk-000/file-000.parquet")
print(df.columns)
# print(df["observation.images"][0])
# for i, action in enumerate(df["action"]):
#     torsorpy = action[28:31]
#     hb       = action[31]
#     vx       = action[32]
#     vy       = action[33]
#     vyaw     = action[34]
#     pyaw     = action[35]
#     if abs(vyaw)>0:
#         print(f"frame {i}:")
#         print(f"  torsorpy={torsorpy}")
#         print(f"  hb={hb}, vx={vx}, vy={vy}, vyaw={vyaw}, pyaw={pyaw}")