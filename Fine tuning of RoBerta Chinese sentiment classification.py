import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline

seed = 5
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
np.random.seed(seed)  # 设置 NumPy 的随机种子
random.seed(seed)  # 设置 Python 自带的随机种子


def load_sentence_polarity(data_path, train_ratio=0.8):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['发布时间', '内容', '日期','标题'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, :]:
            # polar指情感的类别：
            #   ——2：positive
            #   ——1：neutral
            #   ——0：negative
            # sent指对应的句子
            number+=1
            if number%2==0:
                sent = value
            else:
                if value == "正面":
                    polar = 2
                elif value == "负面":
                    polar = 0
                elif value == "中性":
                    polar = 1
            categories.add(polar)
        all_data.append((polar, sent))
    length = len(all_data)
    train_len = int(length * train_ratio)
    train_data = all_data[:train_len]
    test_data = all_data[train_len:]
    return train_data, test_data, categories



class RoBERTaDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 这里可以自行定义，Dataloader会使用__getitem__(self, index)获取数据
        # 这里我设置 self.dataset[index] 规定了数据是按序号取得，序号是多少DataLoader自己算，用户不用操心
        return self.dataset[index]
def sliding_window(sentence, window_size=512, stride=400):

    # 切分后的子句列表
    sub_sentences = []
    # 句子长度
    length = len(sentence)
    # 开始滑动窗口
    start = 0
    while start < length:
        # 计算当前窗口的结束位置
        end = min(start + window_size, length)
        # 切分子句并加入列表
        sub_sentences.append(sentence[start:end])
        # 滑动窗口
        start += stride
    return sub_sentences


def coffate_fn(examples):
    inputs, targets = [], []
    for polar, sent in examples:

        input_text = sent
        if len(sent) > 512:
            input_text = sent[:128] + sent[-382:] #如果句子长度超过512，选择前128个token和后382个toke
            # input_text = sent[:510]  # 如果句子长度不超过512，直接使用
        inputs.append(input_text)
        targets.append(int(polar))

        # 使用tokenizer进行填充
    tokenized_inputs = tokenizer(inputs,
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt",
                                 max_length=512)

    # 确保张量在同一设备上
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    targets = torch.tensor(targets, device=device)

    return tokenized_inputs, targets

# 训练准备阶段，设置超参数和全局变量

#计算验证集损失函数
def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs = {key: value.to(device) for key, value in batch[0].items() if key != "labels"}
            targets = batch[1].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, targets)

            total_loss += loss.item()
        return total_loss / len(dataloader)


batch_size = 3
num_epoch = 2  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./DataSet/中国石油130条数据.csv"  # 数据所在地址
train_ratio = 0.7  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
train_data, test_data, categories = load_sentence_polarity(data_path=data_path, train_ratio=train_ratio)
# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = RoBERTaDataset(train_data)
test_dataset = RoBERTaDataset(test_data)

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=coffate_fn,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)
print(len(test_dataloader),len(test_dataset))
#固定写法，可以牢记，cuda代表Gpu
# torch.cuda.is_available()可以查看当前Gpu是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载预训练的RoBERTa模型和分词器
href="./RoBERTa"
model = AutoModelForSequenceClassification.from_pretrained(href)
tokenizer = AutoTokenizer.from_pretrained(href)



model.to(device)


optimizer = Adam(model.parameters(), learning_rate)  #使用Adam优化器
CE_loss = nn.CrossEntropyLoss()  # 使用crossentropy作为三分类任务的损失函数

# 记录当前训练时间，用以记录日志和存储
timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

# 开始训练，model.train()固定写法
model.train()
train_loss=[]
validate_loss=[]
for epoch in range(1, num_epoch + 1):
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):

        inputs, targets = batch
        inputs = {key: value.to(device) for key, value in inputs.items()}
        targets = targets.to(device)
        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(**inputs)

        logits = bert_output.logits  # 提取logits
        loss = nn.CrossEntropyLoss()(logits, targets)

        # 梯度反向传播
        loss.backward()

        # 根据反向传播的值更新模型的参数
        optimizer.step()

        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += loss.item()
    train_loss.append(total_loss / len(train_dataloader))
    # 在验证集上计算损失函数
    val_loss = compute_loss(model, test_dataloader, CE_loss, device)
    validate_loss.append(val_loss)

#绘损失图
g1=plt.figure()
plt.plot(train_loss,color='r', label='train_loss')
plt.plot(validate_loss,color='g', label='validate_loss')
plt.title('Train/validate Loss')
plt.xlabel('Epoch#')
plt.ylabel('Loss')
plt.yticks([0,0.2,0.4,0.6,0.8,1])
plt.grid()
plt.legend() #添加图例
plt.savefig("./DataSet/微调RoBERTa_LOSS")
plt.show()


#测试过程
# acc统计模型在测试数据上分类结果中的正确个数
acc = 0
tg=0
true=[]
pred=[]


def load_test(data_path):

    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['发布时间', '内容', '日期','标题'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, :]:
            # polar指情感的类别：
            #   ——2：positive
            #   ——1：neutral
            #   ——0：negative
            # sent指对应的句子
            number+=1
            if number%2==0:
                sent = value
            else:
                if value == "正面":
                    polar = 2
                elif value == "负面":
                    polar = 0
                elif value == "中性":
                    polar = 1
            categories.add(polar)
        all_data.append((polar, sent))
    test_data = all_data
    return test_data
data_path = "./DataSet/新闻全文_已标注.csv"  # 数据所在地址
test_data = load_test(data_path=data_path)

test_dataset = RoBERTaDataset(test_data)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)

for batch in tqdm(test_dataloader, desc=f"Testing"):
    inputs, targets = batch
    inputs = {key: value.to(device) for key, value in inputs.items()}
    targets = targets.to(device)

    with torch.no_grad():
        bert_output = model(**inputs)

        # 获取 logits
        logits = bert_output.logits

        # 使用 argmax 获取预测结果
        preds = logits.argmax(dim=1)
        true.append(targets.item())
        pred.append(preds.item())
        acc += (preds== targets).sum().item()
        tg+=len(targets)
# 计算准确率
accuracy=acc/tg
# 计算F1值
f1_micro = f1_score(true, pred, average='weighted')
# 计算召回率
recall = recall_score(true, pred, average='weighted')  # 可以选择其他的 average 参数
# 计算精确度
precision = precision_score(true, pred, average='weighted')  # 可以选择其他的 average 参数
# print(f"召回率: {recall:.4f}\n精确率: {precision:.4f}")
# print(f"准确率: {accuracy:.4f}\nF1值：{f1_micro:.4f}")

#混淆矩阵
labels=["Bearish","Neutral","Bullish"]
label=[0,1,2]
cm = confusion_matrix(true, pred, labels=label)
# 计算每一行的真实样本数
row_sums = cm.sum(axis=1, keepdims=True)
# 将混淆矩阵中的每个元素除以相应的真实样本数，得到概率
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/微调的RoBERTa")
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
print(f"召回率: {recall:.4f}\n精确率: {precision:.4f}")
print(f"准确率: {accuracy:.4f}\nF1值：{f1_micro:.4f}")
print(f"加权F1值:{weighted_f1:.4f}\n")