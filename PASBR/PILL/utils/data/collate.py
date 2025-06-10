from collections import Counter
import numpy as np
import torch as th
import dgl
from torch.nn.utils.rnn import pad_sequence
import math


def label_last(g, last_nid):
    """
    根据输入的序列中最后一个商品获取最后一个商品的one-hot向量
    """
    is_last = th.zeros(g.number_of_nodes(), dtype=th.int32)
    is_last[last_nid] = 1
    g.ndata['last'] = is_last
    return g


def seq_to_eop_multigraph(input_list):
    # 对商品ID进行排序
    seq = input_list[0]
    cate_dict = input_list[1]
    price_dict = input_list[2]
    
    # Initialize sets to track unique missing items
    missing_category_items = set()
    missing_price_items = set()
    
    # TODO:获取原始序列对应的类别序列并进行0填充到长度20
    obtain_cat_seq = []
    for i in seq:
        category = cate_dict.get(str(i))
        if category is None:
            # Use 0 as default category when mapping is missing
            obtain_cat_seq.append(0)
            missing_category_items.add(i)
        else:
            obtain_cat_seq.append(int(category))
    max_lens = 20
    padded_cat_seq = th.zeros(20)
    if len(obtain_cat_seq) < max_lens:
        for j in list(range(len(obtain_cat_seq))):
            padded_cat_seq[j] = obtain_cat_seq[j]
    cat_seq_lens = []
    cat_seq_lens.append(len(obtain_cat_seq))

    # 保留原始顺序
    order_seq = []
    for i in seq:
        if i not in order_seq:
            order_seq.append(i)

    items, index = np.unique(seq, return_inverse=True)

    # 获取商品id --> 索引的转换字典  {原商品id: 索引标签}  如 {123: 0} {232: 1}
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    if len(seq) > 1:
        # 将包含原商品id的序列转成对应的索引序列 {123,232,434} -> {0, 1, 3}
        seq_nid = [iid2nid[iid] for iid in seq]
        src = seq_nid[:-1]
        dst = seq_nid[1:]
    else:
        src = th.LongTensor([])
        dst = th.LongTensor([])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    # 设置当前输入序列的商品ID
    g.ndata['iid'] = th.from_numpy(items)
    # 将当前序列的位置索引也传过去
    index_info = np.arange(1, num_nodes + 1)
    # 计算序列中每个物品与最后一个物品的位置之间的相对位置权重
    # w_i = exp(|i - T| + log2 (T+ 1))
    b = [math.exp((x - max(list(index_info))) + math.log2(max(list(index_info)) + 1)) for x in index_info]
    position_weight_list = np.array([(x / sum(b)) for x in b])
    g.ndata['index_info'] = th.from_numpy(index_info)
    g.ndata['position_weight'] = th.from_numpy(position_weight_list)
    # cid:当前序列对应商品的类别序号,无序
    # pid:当前序列对应商品价格序号,无序
    cid = []
    pid = []
    for i in items:
        category = cate_dict.get(str(i))
        price = price_dict.get(str(i))
        if category is None:
            missing_category_items.add(i)
        if price is None:
            missing_price_items.add(i)
        cid.append(0 if category is None else category)
        pid.append(0 if price is None else price)
    # 保留原始顺序的类别id
    origin_cid = []
    for i in order_seq:
        category = cate_dict.get(str(i))
        if category is None:
            missing_category_items.add(i)
        origin_cid.append(0 if category is None else category)
    
    # Print summary of missing mappings
    total_unique_items = len(items)
    
    # if missing_category_items:
    #     print("\nItems missing category mapping:")
    #     for item in sorted(missing_category_items):
    #         print(f"- Item {item}")
    #     missing_cat_percent = (len(missing_category_items) / total_unique_items) * 100
    #     print(f"Total unique items missing categories: {len(missing_category_items)} ({missing_cat_percent:.2f}% of all items)")
    
    # if missing_price_items:
    #     print("\nItems missing price mapping:")
    #     for item in sorted(missing_price_items):
    #         print(f"- Item {item}")
    #     missing_price_percent = (len(missing_price_items) / total_unique_items) * 100
    #     print(f"Total unique items missing prices: {len(missing_price_items)} ({missing_price_percent:.2f}% of all items)")

    # TODO:将价格信息添加为图上的节点特征
    g.ndata['pid'] = th.from_numpy(np.array(pid).astype(dtype=np.int32))
    g.ndata['cid'] = th.from_numpy(np.array(cid).astype(dtype=np.int32))
    g.ndata['origin_cid'] = th.from_numpy(np.array(origin_cid).astype(dtype=np.int32))

    # 给当前序列转成的图赋上当前序列最后一个商品的one-hot向量
    label_last(g, iid2nid[seq[-1]])
    return g


def seq_to_shortcut_graph(input_list):
    seq = input_list[0]
    items = np.unique(seq)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    num_nodes = len(items)

    # seq_nid: [2, 1, 0, 1]
    seq_nid = [iid2nid[iid] for iid in seq]
    # 两重循环,以序列中每个节点开始,将其后续的每个节点依次组成元组
    # Counter(): 返回一个字典,key=每个元组(即每条边),value=该元组在当前序列中出现的次数
    # counter: Counter({(1, 1): 3, (2, 1): 2, (2, 2): 1, (2, 0): 1, (1, 0): 1, (0, 0): 1, (0, 1): 1})
    counter = Counter(
        [(seq_nid[i], seq_nid[j]) for i in range(len(seq)) for j in range(i, len(seq))]
    )
    # edges: dict_keys([(2, 2), (2, 1), (2, 0), (1, 1), (1, 0), (0, 0), (0, 1)])
    edges = counter.keys()
    # src: (2, 2, 2, 1, 1, 0, 0)  dst: (2, 1, 0, 1, 0, 0, 1)
    src, dst = zip(*edges)

    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g


def collate_fn_factory(*seq_to_graph_fns, cate_dict, price_dict):
    # Initialize counters for tracking progress
    sequences_processed = 0
    total_unique_items_seen = set()
    
    def collate_fn(samples):
        nonlocal sequences_processed, total_unique_items_seen
        seqs, labels = zip(*samples)
        sequences_processed += len(seqs)
        
        # Update total unique items seen
        for seq, _ in samples:
            total_unique_items_seen.update(seq)
            
        if sequences_processed % 1000 == 0:  # Print progress every 1000 sequences
            print(f"\nProcessed {sequences_processed} sequences")
            print(f"Total unique items seen: {len(total_unique_items_seen)}")
            
            # Calculate global statistics for missing mappings
            missing_cats = sum(1 for item in total_unique_items_seen if cate_dict.get(str(item)) is None)
            missing_prices = sum(1 for item in total_unique_items_seen if price_dict.get(str(item)) is None)
            
            # if missing_cats > 0:
            #     missing_cat_percent = (missing_cats / len(total_unique_items_seen)) * 100
            #     print(f"Global missing categories: {missing_cats} ({missing_cat_percent:.2f}% of all unique items)")
            
            # if missing_prices > 0:
            #     missing_price_percent = (missing_prices / len(total_unique_items_seen)) * 100
            #     print(f"Global missing prices: {missing_prices} ({missing_price_percent:.2f}% of all unique items)")

        # 将 商品:类别 映射字典添加到每个输入序列中
        # TODO:将价格信息添加到每个输入序列中
        temp_list = list(seqs)
        for i in range(len(temp_list)):
            temp_list[i] = [temp_list[i], cate_dict, price_dict]
        graph_input = tuple(temp_list)

        inputs = []
        for seq_to_graph in seq_to_graph_fns:
            graphs = list(map(seq_to_graph, graph_input))
            bg = dgl.batch(graphs)
            inputs.append(bg)
        labels = th.LongTensor(labels)
        return inputs, labels

    return collate_fn
