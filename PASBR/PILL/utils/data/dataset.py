import itertools
import numpy as np
import pandas as pd


def create_index(sessions):
    """
    将每条session进行多个子session的划分
    [1, 2, 3, 4] -> [[0, 1], [0, 2], [0, 3]]
    0表示第0条session, 1 2 3表示该session的下一个商品在该session中的索引
    """
    # 获取每条session的长度列表: [3 2 3 16 2]
    lens = np.fromiter(map(len, sessions), dtype=np.int64)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.int64)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None).iloc[:, 0]
    sessions = sessions.apply(lambda x: list(map(lambda y: int(float(y)), x.split(',')))).values
    return sessions


def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items


class AugmentedDataset:
    def __init__(self, sessions, sort_by_length=True):
        self.sessions = sessions
        # 将一条session划分成多条子session及其相应的下一个商品索引(在该session中的索引)
        # [0 1] 表示第0条session, 1表示下一个商品的索引
        # [[ 0  1], [ 0  2], [ 1  1], [ 2  1]...
        index = create_index(sessions)  # columns: sessionId, labelIndex
        if sort_by_length:
            # sort by labelIndex in descending order
            # 所有子session组成一个矩阵,取出该矩阵的第2列(该列表示下一个商品的索引值)
            # 对该列进行降序排列,此处的ind表示在重排后的矩阵中,从上到下每条子session在未排之前的索引
            # 例如[0 1 2 3 4 5] -> [3 1 2 5 4 0]
            # 后者表示下一个商品索引降序排列的子session所在的原始索引列表
            ind = np.argsort(index[:, 1])[::-1]
            # 获取重排后的子session矩阵
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        # 根据输入的id获取指定的子session及其标签
        # [1, 2, 3, 4] -> seq: [1, 2, 3]  label: [4]
        sid, lidx = self.index[idx]
        seq = self.sessions[sid][:lidx]
        label = self.sessions[sid][lidx]
        return seq, label

    def __len__(self):
        return len(self.index)
