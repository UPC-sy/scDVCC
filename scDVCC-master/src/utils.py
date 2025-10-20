import torch
import random
import numpy as np
from sklearn import metrics
from munkres import Munkres

def set_seed(seed=0):#为不同库的随机数生成器设置种子，以确保随机操作的结果是可重复的。
    torch.manual_seed(seed)#设置PyTorch随机种子
    torch.cuda.manual_seed_all(seed)#为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True#设置为True以确保每次运行的结果是相同的
    torch.backends.cudnn.benchmark = False#设置为False以确保每次运行的结果是相同的
    random.seed(seed)#设置Python随机种子
    np.random.seed(seed)#设置Numpy随机种子

def cluster_acc(y_true, y_pred):
    # 将真实标签转换为整数类型
    y_true = y_true.astype(int)

    # 确保标签从0开始
    y_true = y_true - np.min(y_true)

    # 获取真实标签和预测标签的唯一值，并计算类别的数量
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # 如果真实标签和预测标签的类别数量不相等，尝试调整预测标签
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    # 重新计算预测标签的类别数量
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    # 如果真实标签和预测标签的类别数量仍然不相等，打印错误信息并返回
    if numclass1 != numclass2:
        print('n_cluster is not valid')
        return

    # 计算成本矩阵，每个元素表示真实标签中的一个类别与预测标签中的一个类别之间的匹配数量
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # 使用Munkres算法（也称为匈牙利算法）找到成本矩阵的最优匹配
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # 根据最优匹配更新预测标签
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    # 计算新的预测结果的准确性和F1分数
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')

    # 返回准确性和F1分数
    return acc, f1_macro, f1_micro


