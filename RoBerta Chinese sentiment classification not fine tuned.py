import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
href="./RoBERTa"
model = AutoModelForSequenceClassification.from_pretrained(href)
tokenizer = AutoTokenizer.from_pretrained(href)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

data=pd.read_csv("./DataSet/新闻全文_已标注.csv",encoding="UTF-8")
data_length=len(data)
predicted_labels=[]
test_y=data["标签"]
for i in range(0,data_length):
    text=data["新闻全文"][i]
    text_length=len(text)
    if text_length>=512:
        text=text[:128]+text[-382:]
    label = classifier(text)[0]['label']
    print("第"+str(i+1)+"个标签："+label)
    predicted_labels.append(label)
true_labels=[]
for i in range(data_length):
    pred_label=predicted_labels[i]
    true_label=test_y[i]
    if pred_label== "negative":
        predicted_labels[i]=0
    elif pred_label == "neutral":
        predicted_labels[i]=1
    elif pred_label == "positive":
        predicted_labels[i]=2

    if true_label == "负面":
        true_labels.append(0)
    elif true_label == "中性":
        true_labels.append(1)
    elif true_label == "正面":
        true_labels.append(2)
print(type(predicted_labels),type(true_labels))

# 计算准确率
accuracy=sum(1 for x, y in zip(true_labels, predicted_labels) if x == y)/data_length
# 计算F1值
f1_micro = f1_score(true_labels, predicted_labels, average='weighted')
# 计算召回率
recall = recall_score(true_labels, predicted_labels, average='weighted')  # 可以选择其他的 average 参数
# 计算精确度
precision = precision_score(true_labels, predicted_labels, average='weighted')  # 可以选择其他的 average 参数
print(f"召回率: {recall:.4f}\n精确率: {precision:.4f}")
print(f"准确率: {accuracy:.4f}\nF1值：{f1_micro:.4f}")

#混淆矩阵
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(true_labels, predicted_labels, labels=label)
# 计算每一行的真实样本数
row_sums = cm.sum(axis=1, keepdims=True)
# 将混淆矩阵中的每个元素除以相应的真实样本数，得到概率
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/未微调的RoBERTa")
plt.show()

precisions = []  # 3个类别对应的精确度
recalls = []  # 3个类别对应的召回率
weights = [0.5, 0.3, 0.2]  # 负面类别权重最大，正面类别次之，中性类别最小

for i in range(len(label)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precisions.append(precision)
    recalls.append(recall)


def weighted_f1_score(precisions, recalls, weights):
    weighted_precision = sum(p * w for p, w in zip(precisions, weights))
    weighted_recall = sum(r * w for r, w in zip(recalls, weights))
    f1_score = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    return f1_score

weighted_f1 = weighted_f1_score(precisions, recalls, weights)
print(f"加权F1值:{weighted_f1:.4f}\n")