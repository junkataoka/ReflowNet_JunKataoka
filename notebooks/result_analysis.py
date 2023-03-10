#%%
import pandas as pd

# %%
sample1 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test0_notar_all/result_sample1.csv")
sample2 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test1_notar_all/result_sample1.csv")
sample3 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test2_notar_all/result_sample1.csv")
sample_notar = pd.concat([sample1, sample2, sample3], axis=0)
print(sample_notar.mean(axis=0))
print(sample_notar.std(axis=0))

#%%
sample1 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test0_withtar_all/result_sample1.csv")
sample2 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test1_withtar_all/result_sample1.csv")
sample3 = pd.read_csv("/data/home/jkataok1/ReflowNet_ver2_50/models/test2_withtar_all/result_sample1.csv")
sample_notar = pd.concat([sample1, sample2, sample3], axis=0)
print(sample_notar.mean(axis=0))
print(sample_notar.std(axis=0))

# %%
