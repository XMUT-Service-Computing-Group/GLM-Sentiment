import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")
seed = 5
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
np.random.seed(seed)  # 设置 NumPy 的随机种子
random.seed(seed)  # 设置 Python 自带的随机种子
# 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响，请忽略
logging.set_verbosity_error()


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 通过继承nn.Module类自定义符合自己需求的模型
class BertSST2Model(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args:
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        # 类继承的初始化，固定写法
        super(BertSST2Model, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)

        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):

        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        # 将三者输入进模型，如果想知道模型内部如何运作，前面的蛆以后再来探索吧~
        output = self.bert(input_ids, input_tyi, input_attn_mask)


        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def load_sentence_polarity(data_path, train_ratio=0.7):

    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['发布时间', '内容', '日期'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, 1:]:
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




class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):

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
        # 如果句子长度超过512，则进行滑动窗口切分处理
        if len(sent) > 512:
            # # 使用滑动窗口切分句子
            # sub_sentences = sliding_window(sent)
            # inputs.extend(sub_sentences)
            # targets.extend([int(polar)] * len(sub_sentences))

            # 如果句子长度超过512，选择前128个token和后382个token
            input_text = sent[:128] + sent[-382:]
            inputs.append(input_text)
            targets.append(int(polar))
        else:

            inputs.append(sent)
            targets.append(int(polar))

    input_dict = tokenizer(inputs,
                           padding=True,
                           truncation=True,
                           return_tensors="pt",
                           max_length=512)
    targets = torch.tensor(targets)

    return input_dict, targets


# 训练准备阶段，设置超参数和全局变量

#计算验证集损失函数
def compute_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            inputs, targets = [x.to(device) for x in batch]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


batch_size = 4
num_epoch = 2  # 训练轮次
check_step = 1  # 用以训练中途对模型进行检验：每check_step个epoch进行一次测试和保存模型
data_path = "./DataSet/中国石油130条数据.csv"  # 数据所在地址
train_ratio = 0.7  # 训练集比例
learning_rate = 1e-5  # 优化器的学习率

# 获取训练、测试数据、分类类别总数
train_data, test_data, categories = load_sentence_polarity(data_path=data_path, train_ratio=train_ratio)
# 将训练数据和测试数据的列表封装成Dataset以供DataLoader加载
train_dataset = BertDataset(train_data)
test_dataset = BertDataset(test_data)

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

# 加载预训练模型，因为这里是中文数据集，需要用在中文上的预训练模型：bert-base-chinese

pretrained_model_name = 'bert-base-chinese'
# 创建模型 BertSST2Model
model = BertSST2Model(len(categories), pretrained_model_name)

model.to(device)
# 加载预训练模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)


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

        inputs, targets = [x.to(device) for x in batch]

        # 清除现有的梯度
        optimizer.zero_grad()

        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(inputs)

        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        loss = CE_loss(bert_output, targets)

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
plt.savefig("./DataSet/BERT_LOSS")
plt.show()


#测试过程
# acc统计模型在测试数据上分类结果中的正确个数
acc = 0
tg=0
true=[]
pred=[]


def load_test(data_path):
    # 本任务中暂时只用train、test做划分，不包含dev验证集，
    # train的比例由train_ratio参数指定，train_ratio=0.8代表训练语料占80%，test占20%
    all_data = []
    # categories用于统计分类标签的总数，用set结构去重
    categories = set()
    df = pd.read_csv(data_path, encoding="UTF-8")
    data = df.drop(labels=['发布时间', '内容', '日期'], axis=1)

    sent=""
    polar=""
    for i in range(0, len(data)):
        number=0
        for value in data.iloc[i, 1:]:
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

test_dataset = BertDataset(test_data)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             collate_fn=coffate_fn)

for batch in tqdm(test_dataloader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    # with torch.no_grad(): 为固定写法，
    # 这个代码块中的全部有关tensor的操作都不产生梯度。目的是节省时间和空间，不加也没事
    with torch.no_grad():
        bert_output = model(inputs)

        preds=bert_output.argmax(dim=1)
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
plt.savefig("./DataSet/BERT微调")
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
print(f"准确率: {accuracy:.4f}\nF1值：{f1_micro:.4f}")
print(f"加权F1值:{weighted_f1:.4f}\n")
