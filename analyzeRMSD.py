import pandas as pd
from sklearn import metrics

#计算自己训练的和ARES的差别
predicted_rmsd_1 = pd.read_csv('./pdbrms_1gpu_output.csv', usecols=[0,1])
# print(predicted_rmsd_1.head())
ares_rmsd = pd.read_csv('./blind.csv', usecols=[0,2])
ares_rmsd.rename(columns={'tag':'id'}, inplace=True) #将blind.csv的tag列改为id列，使之匹配
# print(ares_rmsd.head())

predicted_rmsd_1['id'] = predicted_rmsd_1['id'].str[:-4] #去掉id中的.pdb使之与blind.csv的id匹配
# print(predicted_rmsd_1.head())

predicted_rmsd = predicted_rmsd_1.merge(ares_rmsd, on=['id']) #方便对比我训练出的与真实ares的区别
predicted_rmsd['-'] = predicted_rmsd['ares']-predicted_rmsd['pred']
# print(predicted_rmsd.head())

predicted_rmsd = predicted_rmsd.dropna(how='any')#去掉nan,否则无法计算MSE
# print(predicted_rmsd.head())
MSE = metrics.mean_squared_error(predicted_rmsd['ares'], predicted_rmsd['pred'])
MAE = metrics.mean_absolute_error(predicted_rmsd['ares'], predicted_rmsd['pred'])
print('MSE=', MSE)
print('MAE=', MAE)
