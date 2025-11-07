pth = '/public/home/202220143416/projects/STAND/logs/supervised_exp_all_test.csv'
pth = '/public/home/202220143416/projects/STAND/logs/supervised_limit_exp_all_test.csv'
import pandas as pd
df = pd.read_csv(pth)
df = df[df['model_name']!='STAND']

dataset_name_order = ['PSM','SWAT','WADI','NIPS_TS_Swan','NIPS_TS_Water']
df = df.groupby(['dataset_name', 'model_name'], sort=False).agg({
    'F1': 'mean',
}).reset_index()

model_list = df['model_name'].unique().tolist()
# 每一个模型是否在所有数据集上都有结果
for model in model_list:
    tmp = df[df['model_name']==model]
    if tmp.shape[0]!=len(dataset_name_order):
        print(f'Model {model} is missing results for some datasets.')
    else:
        print(f'Model {model} has results for all datasets.')