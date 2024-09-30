import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from mordred import Calculator, descriptors
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


def compute_mordred(df):
    """计算Mordred分子描述符"""
    
    data = df.copy()

    calc = Calculator(descriptors)
    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]
    descrs = calc.pandas(mols).astype(float)

    data = pd.concat([data, descrs], axis=1)
    data = data.loc[:, (data != data.iloc[0]).any()] # 去除所有只含有一个值的描述符列
    data.dropna(axis=1, how='any', inplace=True) # 去除所有含缺失值的描述符列
    data.drop(columns=['SMILES'], inplace=True)

    return data


def compute_rdkit(df):
    """计算RDKit分子描述符"""

    data = df.copy()

    mols = [Chem.MolFromSmiles(smi) for smi in data['SMILES']]
    descrs = [Descriptors.CalcMolDescriptors(mol) for mol in mols]
    descrs = pd.DataFrame(descrs)

    data = pd.concat([data, descrs], axis=1)
    data = data.loc[:, (data != data.iloc[0]).any()]
    data.dropna(axis=1, how='any', inplace=True)
    data.drop(columns=['SMILES'], inplace=True)

    return data

def compute_morgan(df):
    """计算Morgan分子指纹"""

    data = df.copy()
    new_columns_data = {}

    for idx in data.index:
        mol = Chem.MolFromSmiles(data.loc[idx, 'SMILES'])
        ECFP_bitinfo = {}
        ECFP = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=ECFP_bitinfo)
        for f in ECFP_bitinfo.keys():
            if f not in new_columns_data:
                new_columns_data[f] = [0] * len(data)
            new_columns_data[f][idx] = len(ECFP_bitinfo[f])

    new_columns_df = pd.DataFrame(new_columns_data)
    new_columns_df.columns = new_columns_df.columns.astype(str)  # 将列名转换为字符串类型
    data = pd.concat([data, new_columns_df], axis=1)
    data = data.fillna(0) # 缺失值用0填充
    data = data.loc[:, (data != data.iloc[0]).any()]
    data.drop(columns=['SMILES'], inplace=True)

    return data

def data_split(df, label='label', test_size=0.2):
    """将数据集分为训练集和测试集（分层采样）"""

    y = df[label]
    X = df.drop(label, axis=1)
    feature_names = X.columns.tolist()

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True, random_state=0)

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test, y_train, y_test


def feature_selection(model, num_features, X_train, y_train, X_test):
    """基于嵌入法选择特征"""
    
    selector = SelectFromModel(model, max_features=num_features).fit(X_train, y_train)
    feature_names = selector.get_feature_names_out().tolist()

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    features_selected = pd.DataFrame(feature_names, columns=['Features'])

    return X_train, X_test, features_selected


def evaluate_model(model, X_train, y_train, X_test, y_test, kf=5, name='', metric='accuracy'):
    """评估二分类模型在训练集和测试集上的性能"""

    val_acc = cross_val_score(model, X_train, y_train, cv=kf, scoring=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    mc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = ['Non-Toxic', 'Toxic']
    conf_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    print(f'{name}\'s results of 10-fold cross-validation are as follows: \n {val_acc} \n')
    print(f'{name}\'s mean result of 10-fold cross-validation is {val_acc.mean():.3g}')
    print(f'{name}\'s Matthews Correlation Coefficient is {mc:.3g} \n')
    print(f'{name}\'s performance on test set is as follows:\n{report}')

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size":10}, cmap="Blues")
    plt.title(f'{name}', fontsize=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()

def evaluate_model_3class(model, X_train, y_train, X_test, y_test, kf=5, name='', metric='f1_macro'):
    """评估三分类模型在训练集和测试集上的性能"""

    val_acc = cross_val_score(model, X_train, y_train, cv=kf, scoring=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    reports = classification_report(y_test, y_pred)
    mc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = ['Non-Toxic', 'Low-Toxic', 'High-Toxic']
    conf_matrix = pd.DataFrame(cm, index=classes, columns=classes)

    print(f'{name}\'s results of 10-fold cross-validation are as follows: \n {val_acc} \n')
    print(f'{name}\'s mean result of 10-fold cross-validation is {val_acc.mean():.3g}')
    print(f'{name}\'s Matthews Correlation Coefficient is {mc:.3g} \n')
    print(f'{name}\'s performance on test set is as follows:\n{reports}')

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size":10}, cmap="Blues")
    plt.title(f'{name}', fontsize=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.show()

def calculate_DT(train_set, k_neighbors, Z=0.5):
    """计算某一个分子与其最近邻分子间的欧几里得距离"""

    DT_values = []
    
    for i, train_sample in enumerate(train_set):
        other_samples = np.delete(train_set, i, axis=0)
        distances = [euclidean(train_sample, other_sample) for other_sample in other_samples]
        sorted_indices = np.argsort(distances)[:k_neighbors]
        nearest_distances = np.array([distances[i] for i in sorted_indices])
        
        gamma_bar = np.mean(nearest_distances)
        sigma = np.std(distances)
        DT = gamma_bar + Z * sigma
        
        DT_values.append(DT)
    
    return DT_values