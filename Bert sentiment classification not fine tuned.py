import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "./DataSet/新闻全文_已标注.csv"
df = pd.read_csv(data_path, encoding="UTF-8")
data = df.drop(labels=['发布时间',"标题", '内容', '日期'], axis=1)
print(data.head())


# 加载BERT预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 初始化列表，用于存储每个文本的所有窗口的特征向量和标签
all_features = []
all_labels = []

# 处理每个长文本
# for text, label in zip(texts, labels):
for i in range(0,len(data)):
    text=data["新闻全文"][i]
    temp_label=data["标签"][i]
    if temp_label=="正面":
        label=2
    elif temp_label=="负面":
        label=0
    else:
        label=1
    # 对于超过512个token的句子，选择前128个token和后382个token
    if len(tokenizer.tokenize(text)) > 512:
        input_text = text[:128] + text[-382:]
    else:
        input_text = text

    # 将处理后的句子转换成模型输入的格式
    inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # 获取BERT模型的文本特征（即最后一层的隐藏状态）
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

    # 使用CLS token作为整体句子的表示（这里假设使用第一个token作为CLS token）
    cls_token = last_hidden_states[:, 0, :].numpy()

    # 添加到特征和标签列表中
    all_features.append(cls_token)
    all_labels.append(label)

# 转换为numpy数组
all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=5)

# 初始化并训练多类别逻辑回归分类器
classifier = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=1000)
classifier.fit(X_train, y_train)

# 预测并评估分类器性能
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


#混淆矩阵
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=label)
# 计算每一行的真实样本数
row_sums = cm.sum(axis=1, keepdims=True)
# 将混淆矩阵中的每个元素除以相应的真实样本数，得到概率
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('y_test labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/BERT未微调")
plt.show()


precisions = []  # 3个类别对应的精确度
recalls = []  # 3个类别对应的召回率
weights = [0.5, 0.3, 0.2]  # 负面类别权重最大，正面类别次之，中性类别最小

for i in range(len(label)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    temp_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    temp_recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precisions.append(temp_precision)
    recalls.append(temp_recall)


# 打印结果
print("Precisions:", precisions)
print("Recalls:", recalls)

def weighted_f1_score(precisions, recalls, weights):
    weighted_precision = sum(p * w for p, w in zip(precisions, weights))
    weighted_recall = sum(r * w for r, w in zip(recalls, weights))
    f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    return f1_score

weighted_f1 = weighted_f1_score(precisions, recalls, weights)
print(f"召回率: {recall:.4f}\n精确率: {precision:.4f}")
print(f"准确率: {accuracy:.4f}\nF1值：{f1:.4f}")
print(f"加权F1值:{weighted_f1:.4f}\n")
