import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
seed = 5
np.random.seed(seed)  # 设置 NumPy 的随机种子
random.seed(seed)  # 设置 Python 自带的随机种子
# pred20=pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRST'))
# feature_data=pred20.columns
# pred20.loc[1,feature_data[0]]=1
# if pred20.loc[1,"A"]==1:
#     print("等于1")
# print(pred20,len(feature_data))


filePath="./DataSet/新闻全文_已标注.csv"
df=pd.read_csv(filePath,encoding="UTF-8")
data=df.drop(labels=['标题','发布时间','内容','日期'],axis=1)

recallss=[]
precisionss=[]
accuracys=[]
f1_micros=[]
weighted_f1s=[]
# paths=["GLM2-1.csv","GLM3-1.csv","GLM2-3.csv","GLM3-3.csv","GLM2-4.csv","GLM3-4.csv","GLM2-5.csv","GLM3-5.csv","GLM2-6.csv","GLM3-6.csv","GLM2-integration.csv","GLM3-integration.csv"]
paths=["GLM3-1-10jis.csv","GLM3-2jis .csv","GLM2-2jis.csv","GLM3-5json (1).csv","GLM2-5jis.csv"]
#
for path in paths:
    pred20=pd.read_csv('News/'+path, encoding='utf-8')
    pred20=pred20.drop(labels=['条目'],axis=1)
    # pred20=pred20.drop(labels=['条目','A','B','C','D','E','F','G','H','I','J'],axis=1)

    #'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T'
    length=len(data)
    pred=pd.DataFrame(pred20)
    # print(pred.iloc[:, :-1])
    df=pred.iloc[:, :-1]
    for i in range(0,length):
        counts = {"正面": 0, "负面": 0, "中性": 0}  # 初始化计数器
        for value in pred20.iloc[i, :-1]:
            if value[0:2]== "正面":
                counts["正面"] += 1
            elif value[0:2] == "负面":
                counts["负面"] += 1
            elif value[0:2] == "中性":
                counts["中性"] += 1
            else:
                continue
        # print(counts)
        if counts["正面"]>counts["负面"] and counts["正面"]>counts["中性"]:
            pred20.loc[i,"标签"]="正面"
        elif counts["负面"]>counts["正面"] and counts["负面"]>counts["中性"]:
            pred20.loc[i,"标签"]="负面"
        else :
            pred20.loc[i,"标签"]="中性"
    predicted_labels=pred20["标签"]
    test_y=data["标签"]

    # 计算准确率
    accuracy=np.sum(data["标签"]==pred20["标签"])/length

    # 计算F1值
    f1_micro = f1_score(test_y, predicted_labels, average='weighted')

    # 计算召回率
    recall = recall_score(test_y, predicted_labels, average='weighted')  # 可以选择其他的 average 参数

    # 计算精确度
    precision = precision_score(test_y, predicted_labels, average='weighted')  # 可以选择其他的 average 参数


    true=[]
    pred=[]
    for i in range(length):
        if data["标签"][i]=="负面":
            true.append(0)
        elif data["标签"][i]=="中性":
            true.append(1)
        elif data["标签"][i]=="正面":
            true.append(2)
        if pred20.loc[i,"标签"]=="负面":
            pred.append(0)
        elif pred20.loc[i,"标签"]=="正面":
            pred.append(2)
        else:
            pred.append(1)
    labels=["Bearish","Neutral","Bullish"]
    label=[0,1,2]

    # #输出预测出错的标签
    # for i in range(len(true)):
    #     if true[i]!=pred[i]:
    #         print(i)

    cm = confusion_matrix(true, pred, labels=label)
    # print("cm:{}".format(cm))

    # 计算每一行的真实样本数
    row_sums = cm.sum(axis=1, keepdims=True)

    # 将混淆矩阵中的每个元素除以相应的真实样本数，得到概率
    cm_prob = cm / row_sums
    ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig("./News/"+path[:-4]+".png")
    plt.show()

    precisions = []  # 3个类别对应的精确度
    recalls = []  # 3个类别对应的召回率
    weights = [0.5, 0.3, 0.2]  # 负面类别权重最大，正面类别次之，中性类别最小

    # 定义计算函数，计算每个类别的真阳性、假阳性和假阴性数量
    def calculate_tp_fp_fn(cm, class_label):
        class_index = labels.index(class_label)
        tp = cm[class_index, class_index]
        fp = np.sum(cm[:, class_index]) - tp
        fn = np.sum(cm[class_index, :]) - tp
        return tp, fp, fn

    def weighted_f1_score(precisions, recalls, weights):
        weighted_precision = sum(p * w for p, w in zip(precisions, weights))
        weighted_recall = sum(r * w for r, w in zip(recalls, weights))

        f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

        return f1_score

    # 计算每个类别的精确度
    def calculate_precision(tp, fp):
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    # 计算每个类别的召回率
    def calculate_recall(tp, fn):
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)

    # 计算每个类别的真阳性、假阳性和假阴性数量
    for class_label in labels:
        tp, fp, fn = calculate_tp_fp_fn(cm, class_label)
        # print(f"类别: {class_label}")
        # print("真阳性数量:", tp)
        # print("假阳性数量:", fp)
        # print("假阴性数量:", fn)

        # 计算每个类别的精确度和召回率
        precision_neg = calculate_precision(tp, fp)
        recall_neg = calculate_recall(tp, fn)

        precisions.append(precision_neg)
        recalls.append(recall_neg)


    # 计算F1值
    weighted_f1 = weighted_f1_score(precisions, recalls, weights)
    print(path)
    print(f"召回率: {recall*100:.2f}%\n精确率: {precision*100:.2f}%")
    print(f"准确率: {accuracy*100:.2f}%\nF1值：{f1_micro*100:.2f}%")
    print(f"加权F1值:{weighted_f1*100:.2f}%\n")
    recallss.append(round(recall*100,2))
    precisionss.append(round(precision*100,2))
    accuracys.append(round(accuracy*100,2))
    f1_micros.append(round(f1_micro*100,2))
    weighted_f1s.append(round(weighted_f1*100,2))

# for i in range(len(weighted_f1s)):
#     print(f"GLM3:{weighted_f1s[i]:.4f}\tGLM2:{weighted_f1s[1+i]:.4f}")

    # # 下面根据混淆矩阵结果的方法也能计算加权F1值
    # precisions = []  # 3个类别对应的精确度
    # recalls = []  # 3个类别对应的召回率
    # for i in range(len(label)):
    #     tp = cm[i, i]
    #     fp = cm[:, i].sum() - tp
    #     fn = cm[i, :].sum() - tp
    #
    #     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    #     recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    #
    #     precisions.append(precision)
    #     recalls.append(recall)
    # weighted_f1 = weighted_f1_score(precisions, recalls, weights)
    # print(f"加权值:{weighted_f1:.4f}\n")

















# print(precisionss)
# print(recallss)
#
# print(accuracys)
#
# print(f1_micros)
#
# print(weighted_f1s)

# # 设置柱状图的宽度
# bar_width = 0.5
# bar_width_total = 2.5  # 总的柱状图宽度
#
# # 设置柱状图的位置
# models=["GLM3-1","GLM2-1","GLM3-3","GLM2-3","GLM3-4","GLM2-4","GLM3-5","GLM2-5","GLM3-6","GLM2-6","GLM3-integration","GLM2-integration"]
# rr=[3,6,9,12,15,18,21,24,27,30,33,36]
# r1 = [x - bar_width*2 for x in rr]
# r2 = [x - bar_width for x in rr]
# r3 = [x  for x in rr]
# r4 = [x + bar_width for x in rr]
# r5 = [x + bar_width*2 for x in rr]
#
# # RGB 转换为 Hex
# def rgb_to_hex(r, g, b):
#     return '#{:02x}{:02x}{:02x}'.format(r, g, b)
#
# precision_color = rgb_to_hex(140, 205, 191)
# recall_color = rgb_to_hex(205, 224, 165)
# accuracy_color = rgb_to_hex(249, 219, 149)
# f1_micro_color = rgb_to_hex(239, 123, 118)
# weighted_f1_color = rgb_to_hex(197, 168, 206)
#
# plt.yticks(np.arange(35, 80, 10))
#
#
# # 创建柱状图
# plt.figure(figsize=(14, 8))
# plt.bar(r1, recallss, color=recall_color, width=bar_width, edgecolor='grey', label='Recall')
# plt.bar(r2, precisionss, color=precision_color, width=bar_width, edgecolor='grey', label='Precision')
# plt.bar(r3, accuracys, color=accuracy_color, width=bar_width, edgecolor='grey', label='Accuracy')
# plt.bar(r4, f1_micros, color=f1_micro_color, width=bar_width, edgecolor='grey', label='F1_score')
# plt.bar(r5, weighted_f1s, color=weighted_f1_color, width=bar_width, edgecolor='grey', label='Weighted_F1')

# # 设置 y 轴的范围
# plt.ylim(35, 80)  # 将 y 轴的开始坐标设置为 0.5，结束坐标设置为 1.0
#
# # 添加标签
# plt.xlabel('Models', fontweight='bold')
# plt.ylabel('Scores', fontweight='bold')
# plt.xticks(rr, models)
# plt.title('Comparison of model Performance Metrics')

# # 显示每个柱状图的高度
# for r, precision, recall, accuracy, f1_micro, weighted_f1 in zip(r1, precisionss, recallss, accuracys, f1_micros, weighted_f1s):
#     plt.text(r, recall, '{:.2f}'.format(recall), ha='center', va='bottom', rotation=90)
#     plt.text(r + bar_width, precision, '{:.2f}'.format(precision), ha='center', va='bottom', rotation=90)
#     plt.text(r + 2 * bar_width, accuracy, '{:.2f}'.format(accuracy), ha='center', va='bottom', rotation=90)
#     plt.text(r + 3 * bar_width, f1_micro, '{:.2f}'.format(f1_micro), ha='center', va='bottom', rotation=90)
#     plt.text(r + 4 * bar_width, weighted_f1, '{:.2f}'.format(weighted_f1), ha='center', va='bottom', rotation=90)

# 显示图例
# plt.legend()
# plt.grid(ls=":",color="gray",alpha=0.5)
# 显示图表
# plt.show()