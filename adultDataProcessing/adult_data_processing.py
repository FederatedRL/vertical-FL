import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier

training_data = r'data\adult.data'
test_data = r'data\adult.test'
columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'EdNum', 'MaritalStatus',
           'Occupation', 'Relationship', 'Race', 'Sex', 'CapitalGain',
           'CapitalLoss', 'HoursPerWeek', 'Country', 'Income']

df_train_set = pd.read_csv(training_data, names=columns)
df_test_set = pd.read_csv(test_data, names=columns, skiprows=1)


df_train_set.drop('fnlgwt', axis=1, inplace=True)  # 删除fnlgwt列
df_test_set.drop('fnlgwt', axis=1, inplace=True)
# =============================================================================
# 数据清洗
for i in df_train_set.columns:
    df_train_set[i].replace('?', 'Unknown', inplace=True)
    df_test_set[i].replace('?', 'Unknown', inplace=True)
    for col in df_train_set.columns:
        if df_train_set[col].dtype != 'int64':
            df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(" ", ""))
            df_train_set[col] = df_train_set[col].apply(lambda val: val.replace(".", ""))
            df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(" ", ""))
            df_test_set[col] = df_test_set[col].apply(lambda val: val.replace(".", ""))

# Eduction 和 EdNum特征相似，可以删除Eduction
# Country对收入影响不大，也可以删除
df_train_set.drop(["Country", "Education"], axis=1, inplace=True)
df_test_set.drop(["Country", "Education"], axis=1, inplace=True)

# Age 和 EdNum 列是数值型的，将连续数值型转换为更高效的方式，
# 比如将年龄转换为10的整数倍，教育年限转换为5的整数倍
colnames = list(df_train_set.columns)  # 将数据的表头转换为列表并储存
colnames.remove('Age')
colnames.remove('EdNum')
colnames = ['AgeGroup', 'EduGroup'] + colnames
labels = ["{0}-{1}".format(i, i + 9) for i in range(0, 100, 10)]

df_train_set['AgeGroup'] = pd.cut(df_train_set.Age, range(0, 101, 10), right=False, labels=labels)
#把一组数据分割成离散的区间
df_test_set['AgeGroup'] = pd.cut(df_test_set.Age, range(0, 101, 10), right=False, labels=labels)

labels = ["{0}-{1}".format(i, i + 4) for i in range(0, 20, 5)]
df_train_set['EduGroup'] = pd.cut(df_train_set.EdNum, range(0, 21, 5), right=False, labels=labels)
df_test_set['EduGroup'] = pd.cut(df_test_set.EdNum, range(0, 21, 5), right=False, labels=labels)

df_train_set = df_train_set[colnames]  # 仅提取包含colnames中的列，且按照colnames排序
df_test_set = df_test_set[colnames]
print(df_train_set[0:1])
# =============================================================================
# 将非数值型数据转换为数值型数据
# df_train_set.Income.value_counts()
# df_test_set.Income.value_counts()
mapper = DataFrameMapper([('AgeGroup', LabelEncoder()), ('EduGroup', LabelEncoder()),
                          ('Workclass', LabelEncoder()), ('MaritalStatus', LabelEncoder()),
                          ('Occupation', LabelEncoder()), ('Relationship', LabelEncoder()),
                          ('Race', LabelEncoder()), ('Sex', LabelEncoder()),
                          ('Income', LabelEncoder())], df_out=True, default=None)

cols = list(df_train_set.columns)
cols.remove('Income')
cols = cols[:-3] + ['Income'] + cols[-3:]  # 将Income列转移到中间去

df_train = mapper.fit_transform(df_train_set.copy())
df_train.columns = cols
print(df_train.loc[0])
df_test = mapper.transform(df_test_set.copy())
df_test.columns = cols

cols.remove('Income')

# 训练数据与测试数据划分
x_train, y_train = df_train[cols].values, df_train['Income'].values
x_test, y_test = df_test[cols].values, df_test['Income'].values
# =============================================================================
# 模型初步训练与评分
treeClassifier = DecisionTreeClassifier()
treeClassifier.fit(x_train, y_train)
score = treeClassifier.score(x_test, y_test)
print('The score before GridSearch:', score)
