import random
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

random.seed = 16

data = pd.read_csv('ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')

stop_words = []
with open('stopwords.txt', encoding='UTF-8') as f:
    for line in f.readlines():
        line = line.strip()
        stop_words.append(line)


def seg_word(doc):
    seg_list = jieba.cut(doc)
    return list(seg_list)


def filter_map(arr):
    """
    去除字符串的停用词
    :param arr: 一个句子
    :return:
    """
    res = ""
    for c in arr:
        if c not in stop_words and c != ' ' and c != '\xa0'and c != '\n' and c != '\ufeff' \
                and c != '\r':
            res += c
    return res


def filter_char_map(arr):
    res = []
    for c in arr:
        if c not in stop_words and c != ' ' and c != '\xa0'and c != '\n' and c != '\ufeff' \
                and c != '\r':
            res.append(c)
    return " ".join(res)


def get_char(arr):
    res = []
    for c in arr:
        res.append(c)
    return list(res)


# print(data.content)
data.content = data.content.map(lambda x: filter_map(x))
data.content = data.content.map(lambda x: get_char(x))
print(data["content"])
data.to_csv("preprocess/train_char.csv", index=None)

# 训练词向量
line_sent = []
for s in data["content"]:
    line_sent.append(s)
word2vec_model = Word2Vec(line_sent, size=100, window=10, min_count=1, workers=4, iter=15)
word2vec_model.wv.save_word2vec_format("word2vec/chars.vector", binary=True)

# 处理验证数据集
validation = pd.read_csv("ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv")
validation.content = validation.content.map(lambda x: filter_map(x))
validation.content = validation.content.map(lambda x: get_char(x))
print(validation["content"])
validation.to_csv("preprocess/validation_char.csv", index=None)

# 处理测试数据集
test = pd.read_csv("ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv")
test.content = test.content.map(lambda x: filter_map(x))
test.content = test.content.map(lambda x: get_char(x))
test.to_csv("preprocess/test_char.csv", index=None)
