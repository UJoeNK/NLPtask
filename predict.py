import json
import random
import pickle
from TextClassification import TextClassification

'''
从样例数据中抽取10条进行演示。模型是读取训练好的模型，label是原标签，predict是预测标签。
'''

r = []
for i in range(10):
    r.append(random.randint(1, 10000))

with open('./TextClassification/data/data_sample.json', mode='r', encoding='utf8') as f:
    data_raw = f.readlines()
data = [json.loads(data_raw[i]) for i in r]

print('data: ', data)

test = [i['fact'] for i in data]
label = [i['accusation'] for i in data]

clf = TextClassification()
with open('./model.pkl', 'rb') as f:
    clf = pickle.load(f)
predict = clf.predict(test)
predict = clf.label2tag(predict, clf.preprocess.label_set)

print('label: ', label)
print('predict: ', predict)
