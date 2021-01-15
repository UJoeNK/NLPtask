import json
import pandas as pd
import os

localpath = os.path.dirname(__file__)


def load_data():
    with open(localpath + '/data/data.json', mode='r', encoding='utf8') as f:
        data_raw = f.readlines()
    data = [json.loads(i) for i in data_raw]
    return data
