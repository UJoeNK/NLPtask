from TextClassification import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import json
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

# 导入数据，拆分训练集和测试集
if os.path.exists("x_train.json"):
    print("data exists.")
    x_train = json.load(open("x_train.json", "r", encoding="utf8"))
    y_train = json.load(open("y_train.json", "r", encoding="utf8"))
    x_test = json.load(open("x_test.json", "r", encoding="utf8"))
    y_test = json.load(open("y_test.json", "r", encoding="utf8"))
else:
    data = load_data()
    x = [i['fact'] for i in data]
    y = [i['accusation'] for i in data]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    json.dump(x_train, open("x_train.json", "w", encoding="utf8"))
    json.dump(x_test, open("x_test.json", "w", encoding="utf8"))
    json.dump(y_train, open("y_train.json", "w", encoding="utf8"))
    json.dump(y_test, open("y_test.json", "w", encoding="utf8"))

##### 以下是训练过程 #####

from TextClassification import TextClassification

clf = TextClassification()
texts_seq, texts_labels = clf.get_preprocess(x_train, y_train, word_len=1, num_words=10000, sentence_len=256)
clf.fit(texts_seq, texts_labels, 16, 512)

# 保存整个模块,包括预处理和神经网络
with open('./model.pkl', 'wb') as f:
    pickle.dump(clf, f)

##### 以下是预测过程 #####

# 导入刚才保存的模型
with open('./model.pkl', 'rb') as f:
    clf = pickle.load(f)
y_predict = clf.predict(x_test)
y_predict = clf.label2tag(y_predict, clf.preprocess.label_set)
score = sum([y_predict[i] == y_test[i] for i in range(len(y_predict))]) / len(y_predict)
print(score)
