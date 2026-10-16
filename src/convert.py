import pandas as pd, numpy as np, time
t=time.time()
for name, fn in [("train","dataset/train (1).csv"),("test","dataset/test (1).csv"),("sample","dataset/sample_submission (1).csv")]:
    df = pd.read_csv(fn, engine="pyarrow")
    print(name, df.shape, time.time()-t)
    df.to_parquet(f"cache/{name}.parquet", index=False)
