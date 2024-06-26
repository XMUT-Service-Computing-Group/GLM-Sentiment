import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import re
import random
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
seed = 5
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
np.random.seed(seed)  # 设置 NumPy 的随机种子
random.seed(seed)  # 设置 Python 自带的随机种子
from pylab import mpl

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()
print("正在运行")
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

print("label")
filePath = "./DataSet/新闻全文_已标注.csv"
df = pd.read_csv(filePath, encoding="UTF-8")
data = df.drop(labels=['发布时间', '内容', '日期'], axis=1)
name = "GLM3-5json"
pred20 = pd.DataFrame(columns=list('ABCDEFGHIJKLMNOPQRST'))
# ABCDEFGHIJKLMNOPQRST
features = pred20.columns
feature_length = len(features)
data_length = len(data)
# 使用GLM进行20次问答
for i in range(0, feature_length):
    print("第" + str(i + 1) + "次提问")
    for j in range(0, data_length):
        background = "中国石油化工股份有限公司的产品主要包括原油、天然气、化纤、化肥、橡胶、成品油等。其竞争对手主要包括其他大型国有石油和化工企业，以及一些国际上的能源和化工巨头。金融新闻对股票市场的影响是复杂而多样的。新闻中的积极或消极或中性信息会引发投资者的情绪波动。\
        积极的新闻能提高投资者的信心，促使他们更倾向于购买股票，从而推动股价上涨；\
        消极的新闻则会引发恐慌情绪，导致投资者纷纷抛售股票，进而导致股价下跌；\
        中性的新闻则没有明显的积极或消极情绪倾向，被投资者视为市场的常态或者暂时性的平静状态，但投资者会将这些信息作为参考做出投资决策。在这种情况下，投资者会保持观望态度，维持现有的持仓或采取谨慎的交易策略，而不会急于大规模买入或抛售股票，在一定程度上维持市场的平稳状态。\
        同时，融资的买入消息和卖出消息在金融市场中具有重要意义。\
        当融资买入规模较大、用途有利于公司发展或市场情绪乐观时，通常被视为积极消息，可能推动股价上涨。\
        相反，如果融资买入规模较小、用途不利于公司发展或市场情绪悲观，可能被视为消极消息，导致股价下跌。\
        而当融资买入规模适中、用途一般或市场整体趋势稳定时，融资的买入和卖出可能被视为中性消息，股价波动较为平稳。\
        当融资的卖出规模较大、用途符合公司战略或市场情绪乐观时，可能被视为积极消息，因为这可能意味着公司获得了资金用于扩张或投资，从而提升投资者信心，推动股价上涨。\
        相反，如果融资的卖出规模较小、用途不明确或市场情绪悲观，可能被视为消极消息，可能引发投资者对公司未来发展的担忧，导致股价下跌。\
        当融资的卖出规模适中、用途一般或市场整体趋势稳定时，融资的卖出可能被视为中性消息，股价波动较为平稳。"
        content = "知识背景：" + background + "以下是三个样本：\
        新闻内容：'''3月25日，中国石化（600028）举行2023年度业绩说明会。中国石化董事长马永生回应中国石化2023年净利润出现下滑称，2023年，国际原油价格总体呈震荡下行走势，同比下降18.4%，境内化工市场供过于求、毛利仍处于低位。面对挑战，公司全方位优化生产经营，大力协同攻坚创效，强化成本费用管控，推动全产业链提质增效，取得了来之不易的经营成果。马永生表示，公司净利润同比下降，主要是受库存变动减利以及计提矿业权出让收益等影响。若将以上因素剔除，同口径利润是大幅改善的。'''，分析：\
        {\
        主体：中国石化\
        事件：3月25日，中国石化举行2023年度业绩说明会，董事长马永生回应2023年净利润下滑情况。\
        影响：尽管净利润同比下降，但马永生解释了下降原因并提到公司在优化生产经营和成本管控方面取得的成果。若剔除一些不利因素，公司的同口径利润有明显改善。\
        情绪：中性\
        }；\
       新闻内容： '''证券之星消息，中国石化2023年年报显示，公司主营收入32122.15亿元，同比下降3.19%；归母净利润604.63亿元，同比下降9.87%；扣非净利润602.34亿元，同比上升3.92%；其中2023年第四季度，公司单季度主营收入7422.74亿元，同比下降14.17%；单季度归母净利润74.97亿元，同比下降23.79%；单季度扣非净利润99.55亿元，同比上升417.14%；负债率52.7%，投资收益58.11亿元，财务费用99.22亿元，毛利率15.65%。'''，分析：\
        {\
        主体： 中国石化\
        事件： 中国石化发布2023年年报，显示公司主营收入和归母净利润同比下降，扣非净利润同比上升；第四季度主营收入和归母净利润同比下降，扣非净利润同比大幅上升。\
        影响： 尽管扣非净利润有所上升，但主营收入和归母净利润的下降反映出公司面临的经营压力和挑战。此外，较高的负债率也可能给公司未来的财务状况带来负面影响。总体来看，年报反映出公司在当前市场环境下的经营困境。\
        情绪： 负面\
        }；\
        新闻内容：'''3月13日，中国石化与宁德时代新能源科技股份有限公司在北京签署战略合作框架协议。双方表示，将进一步深化战略合作，拓宽合作领域，延伸产业链条，加快转型升级步伐，推动双方合作迈上新台阶。'''，分析：\
        {\
        主体： 中国石化\
        事件： 中国石化与宁德时代新能源科技股份有限公司在北京签署战略合作框架协议。\
        影响： 此次合作协议的签署表明两家公司将进一步深化战略合作，拓宽合作领域，延伸产业链条，加快转型升级步伐，这有望推动双方在各自领域实现更大的发展和创新，提升市场竞争力和投资者信心。\
        情绪： 正面\
        }。\
        请你扮演金融新闻情绪分类的专家，参考上述样本案例，结合上文提供的知识背景按照jis格式（主体、事件、影响、情绪）分析下文三引号中的新闻内容传达了什么情绪？是正面、负面还是中性的？如果无法判断新闻内容表达的情绪，默认答案为中性。输出jis格式如下：\
        {\
        主体：中国石化\
        事件：\
        影响：\
        情绪：\
        }\
        新闻内容：‘’‘" + data["新闻全文"][j] + "’‘’，分析："

        response, history = model.chat(tokenizer, content, history=[])
        # 使用正则表达式匹配“情绪”字段
        emotion = re.search(r'情绪：(.+)', response)
        if emotion == None:
            emotion = re.search(r'情绪:(.+)', response)
        if emotion != None:
            pred20.loc[j, features[i]] = emotion.group(1).strip()
        else:
            pred20.loc[j, features[i]] = "中性"
        if j % 40 == 0:
            print("第" + str(j) + "条数据")
            print(response, emotion)

# filePath="/kaggle/input/full-text-news/_.csv"
# df=pd.read_csv(filePath,encoding="UTF-8")
# data=df.drop(labels=['发布时间','内容','日期'],axis=1)
# pred20=pd.read_csv('/kaggle/input/pred-label/pred.csv',encoding='utf-8')

# import sys
sys.path.append("../output")
pred20 = pred20.reset_index(drop=True)
pred20.to_csv('/kaggle/working/' + name + '.csv', index_label="条目")
# print("label")


for i in range(0,data_length):
    counts = {"正面": 0, "负面": 0, "中性": 0}  # 初始化计数器
    for value in pred20.iloc[i, :-1]:
        value=value[:2]
#         print(value)
        if value== "正面" or value=="上涨"  or value=="积极":
            counts["正面"] += 1
        elif value == "负面" or value=="下跌":
            counts["负面"] += 1
        elif value == "中性" or value=="融资" or value=="无法" or value=="稳定":
            counts["中性"] += 1
    if counts["正面"]>counts["负面"] and counts["正面"]>counts["中性"]:
        pred20.loc[i,"标签"]="正面"
    elif counts["负面"]>counts["正面"] and counts["负面"]>counts["中性"]:
        pred20.loc[i,"标签"]="负面"
    else :
        pred20.loc[i,"标签"]="中性"
predicted_labels=pred20["标签"]
test_y=data["标签"]
# 计算准确率
accuracy=np.sum(data["标签"]==pred20["标签"])/data_length
# 计算F1值
f1_micro = f1_score(test_y, predicted_labels, average='weighted')
# 计算召回率
recall = recall_score(test_y, predicted_labels, average='weighted')  # 可以选择其他的 average 参数
# 计算精确度
precision = precision_score(test_y, predicted_labels, average='weighted')  # 可以选择其他的 average 参数
print(f"召回率: {recall:.4f}\n精确率: {precision:.4f}")
print(f"准确率: {accuracy:.4f}\nF1值：{f1_micro:.4f}")

true=[]
pred=[]
for i in range(data_length):
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
# 计算每一行的真实样本数
row_sums = cm.sum(axis=1, keepdims=True)
# 将混淆矩阵中的每个元素除以相应的真实样本数，得到概率
cm_prob = cm / row_sums
ax = sns.heatmap(cm_prob, annot=True, fmt=".2", cmap="Blues", xticklabels=labels, yticklabels=labels)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.savefig("./DataSet/新闻全文_已标注.png")
plt.show()