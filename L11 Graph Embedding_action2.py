import pandas as pd
import numpy as np
import warnings
import missingno as msno
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 数据加载与数据探索EDA
train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
train_data.head()

test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
test_data.head()

print(train_data.isnull().any())
print(train_data.isnull().sum())

print(test_data.isnull().any())
print(test_data.isnull().sum())

print(train_data.info())
print(test_data.info())

# 注册时间探索
dates = pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce')
min_date = pd.to_datetime('19910101', format='%Y%m%d')

train_data['regTime'] = (dates - min_date).dt.days
test_data['regTime'] = (pd.to_datetime(train_data['regDate'], format='%Y%m%d', errors='coerce') - min_date).dt.days
print(train_data.head())

# 汽车使用时间
train_data['usedTime'] = train_data['creatTime'] - train_data['regTime']
test_data['usedTime'] = test_data['creatTime'] - test_data['regTime']
print(train_data.head())

# Sample随机抽取
sample = train_data.sample(1000)
msno.matrix(sample)
plt.show()
msno.bar(sample)
plt.show()
msno.heatmap(sample)
plt.show()

# 选择特征列
feature_cols = [col for col in numerical_cols if col not in ['SaleID','price']]
print(feature_cols)

numerical_cols = train_data.select_dtypes(exclude='object').columns
print(numerical_cols)

# 特征提取
X_data = train_data[feature_cols]
Y_data = train_data['price']
X_test = test_data[feature_cols]

# 定义统计函数
def show_stats(data):
    print('min :', np.min(data))
    print('max :', np.max(data))
    print('ptp :', np.ptp(data))
    print('mean:', np.mean(data))
    print('std :', np.std(data))
    print('var :', np.var(data))

# 统计标签的基本分布信息
print('Price的统计情况:')
show_stats(Y_data)

# 缺失值补全
X_data = X_data.fillna(-1)
X_test = X_test.fillna(-1)

# 神经网络进行预测
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
X = np.array(X_data)
y = np.array(Y_data).reshape(-1,1)
X_ = np.array(X_test)
X.shape, y.shape, X_.shape

# 数据规范
ss = MinMaxScaler()
X = ss.fit_transform(X)
X_ = ss.transform(X_)

# 切分数据集
x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.3)


model = keras.Sequential([
        keras.layers.BatchNormalization(),
        keras.layers.Dense(250,activation='relu',input_shape=[X.shape[1]]), 
        keras.layers.Dense(250,activation='relu'), 
        keras.layers.Dense(250,activation='relu'), 
        keras.layers.Dense(1)])
model.compile(loss='mean_absolute_error', optimizer='Adam')

# 训练
model.fit(x_train,y_train,batch_size = 2048,epochs=350)

#输出结果
y_=model.predict(X_)
show_stats(y_)
data_test_price = pd.DataFrame(y_,columns = ['price'])
results = pd.concat([test_data['SaleID'],data_test_price],axis = 1)
results.to_csv('used_car_price_forecasting.csv',sep = ',',index = None)