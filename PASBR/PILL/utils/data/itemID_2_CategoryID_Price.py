#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Author   : Sword
 @Date     : 2021/2/18 0018 
 @Time     : 下午 23:59
 @Version  : v1.0
 @File     : itemID_2_CategoryID_Price.py 
 @Describe :
"""
import numpy as np
import pandas as pd


def itemid_2_categoryid():
    """
    三表遍历,太花时间
    """
    with open("../../datasets/diginetica/train_before.txt", 'r') as seq_f:
        lines = seq_f.readlines()
        # 存放每一个session对应的类别id列表
        total_cate_list = []
        no = 1
        for line in lines:
            if no % 100 == 0:
                print("共{}行, 正在处理第{}行".format(len(lines), no))
            line_2_id_list = line.split(',')
            with open("../../new_oid2nid.csv", 'r') as o2nid_f:

                # 获取ID对照表的所有行
                oid2nid_lines = o2nid_f.readlines()
                # 遍历session中的每个itemID
                # 取出一行itemID进行遍历匹配,并转成对应的类别ID列表
                o_id_list = []
                for each_id in line_2_id_list:
                    # 针对ID对照表的每一行
                    for each_id_line in oid2nid_lines:
                        id_line_2_list = each_id_line.split(',')
                        # 根据新ID找到原始ID
                        if each_id.strip() == id_line_2_list[1].strip():
                            o_id = id_line_2_list[0]
                            o_id_list.append(o_id)
                match_category_id = match_category(o_id_list)
                total_cate_list.append(match_category_id)
                no += 1
        for each_cate_list in total_cate_list:
            # print(each_cate_list)
            write_category_seq(each_cate_list)


def itemid_2_categoryid_fast():
    # oid: category_id
    cate_dict = {}

    # nid: oid
    nid2oid = {}
    with open("../../datasets/product-categories2.csv", 'r') as cate_f:
        # 获取类别对照表的所有行
        cate_lines = cate_f.readlines()
        for each_cate_line in cate_lines:
            cate_line_2_list = each_cate_line.split(';')
            cate_dict[cate_line_2_list[0]] = cate_line_2_list[1].strip()

    with open("../../new_oid2nid.csv", 'r') as o2nid_f:

        # 获取ID对照表的所有行
        oid2nid_lines = o2nid_f.readlines()
        # 遍历session中的每个itemID
        # 取出一行itemID进行遍历匹配,并转成对应的类别ID列表
        for each_id_line in oid2nid_lines:
            id_line_2_list = each_id_line.split(',')
            # 根据新ID找到原始ID
            nid2oid[id_line_2_list[1].strip()] = id_line_2_list[0]

    with open("../../datasets/diginetica/train_before.txt", 'r') as seq_f:
        lines = seq_f.readlines()
        # 存放每一个session对应的类别id列表
        total_cate_list = []
        no = 1
        for line in lines:
            if no % 100 == 0:
                print("共{}行, 正在处理第{}行".format(len(lines), no))
            line_2_id_list = line.split(',')
            # 存放每一个session对应的类别id
            each_line_cateid_list = []
            # 遍历每一个session,获取其对应的类别id列表
            for each_id in line_2_id_list:
                # nid -> oid
                get_oid = nid2oid.get(each_id.strip())
                # oid -> cate_id
                get_category_id = cate_dict.get(get_oid)
                each_line_cateid_list.append(get_category_id)
            total_cate_list.append(each_line_cateid_list)
            no += 1
        for each_cate_list in total_cate_list:
            # print(each_cate_list)
            write_category_seq(each_cate_list)


def del_line():
    # 删除csv文件中的空行
    with open("oid2nid_category.csv", 'r') as f:
        lines = f.readlines()
        with open("../../new_oid2nid_category.csv", 'w') as f1:
            for line in lines:
                if len(line) == 1:
                    continue
                f1.writelines(line)


def half_train_seq():
    # 将train.txt分成两份
    with open("../../datasets/diginetica/train.txt", 'r') as f:
        lines = f.readlines()
        i = 94318
        with open("../../datasets/diginetica/train_after.txt", 'w') as f1:
            for line in lines[94318:]:
                if i == 188637:
                    break
                f1.writelines(line)
                i += 1


def match_category(oid_list):
    """
    根据输入的商品ID列表获取对应的类别ID列表
    """
    match_category_id = []
    with open("../../datasets/product-categories2.csv", 'r') as cate_f:
        # 获取类别对照表的所有行
        cate_lines = cate_f.readlines()
        for oid in oid_list:
            for each_cate_line in cate_lines:
                cate_line_2_list = each_cate_line.split(';')
                if oid == cate_line_2_list[0]:
                    match_category_id.append(cate_line_2_list[1].strip())
    return match_category_id


def write_category_seq(cate_list):
    # 将输入的list写入到文件中的一行
    with open("niid_2_ncid.csv", 'a') as f:
        i = 0
        for b in cate_list:
            f.write(b)
            if i != len(cate_list) - 1:
                f.write(',')
                i += 1
        f.write('\n')


def train_cate_seq_2_format():
    """
    将类别序列转成标准数据集格式: sessionID;categoryID
    """
    with open("train_cate_seq.txt", 'r') as start_f:
        lines = start_f.readlines()

        # 假设第一行的序列对应sessionID=1
        cur_line_index = 1
        with open("train_cate_seq_format.csv", 'w') as write_f:
            for line in lines:
                if cur_line_index % 100 == 0:
                    print("共{}行,正在处理第{}行".format(len(lines), cur_line_index))
                line_2_list = line.split(',')
                for each_cateid in line_2_list:
                    content = str(cur_line_index) + ";" + each_cateid.strip()
                    write_f.write(content + '\n')
                cur_line_index += 1


def update_category_id_2_train():
    """
    更新类别ID: 从0开始
    最后重新保存成类别序列
    """
    import pandas as pd
    import csv
    df = pd.read_csv('train_cate_seq_format.csv', delimiter=';')
    cateid_new, uniques = pd.factorize(df.categoryId)
    df = df.assign(categoryId=cateid_new)
    oid2nid = {oid: i for i, oid in enumerate(uniques)}
    # 将新旧类别id对照表保存下来
    # 将新旧ID的对应关系字典保存成oid2nid.csv
    with open("oid2nid_category.csv", 'w') as csv_f:
        writer = csv.writer(csv_f)
        for key, value in oid2nid.items():
            writer.writerow([key, value])
    category_seq = df.groupby("sessionId").categoryId.apply(lambda x: ','.join(map(str, x)))
    category_seq.to_csv('new_categoryId_seq.txt', sep='\t', header=False, index=False)


def new_itemID_2_new_categoryID():
    """
    获取新商品ID与新商品类别ID的映射文件
    新商品ID  :  新类别ID
    """
    # nid: oid
    nid2oid = {}

    # oid: category_id
    cate_dict = {}

    # o_category_id : n_category_id
    oid2nid_category_dict = {}
    with open("../../datasets/product-categories2.csv", 'r') as cate_f:
        # oid : o_category_id
        # 获取类别对照表的所有行
        cate_lines = cate_f.readlines()
        for each_cate_line in cate_lines:
            cate_line_2_list = each_cate_line.split(';')
            cate_dict[cate_line_2_list[0]] = cate_line_2_list[1].strip()

    with open("../../new_oid2nid.csv", 'r') as o2nid_f:
        # nid : oid
        # 获取ID对照表的所有行
        oid2nid_lines = o2nid_f.readlines()
        for each_id_line in oid2nid_lines:
            id_line_2_list = each_id_line.split(',')
            # 根据新ID找到原始ID
            nid2oid[id_line_2_list[1].strip()] = id_line_2_list[0]

    with open("new_oid2nid_category.csv", 'r') as o2nid_category_f:
        # o_category_id : n_category_id
        oid2nid_category_lines = o2nid_category_f.readlines()
        for each_line in oid2nid_category_lines:
            each_id_line_2_list = each_line.split(',')
            oid2nid_category_dict[each_id_line_2_list[0]] = each_id_line_2_list[1].strip()

    with open("../../datasets/diginetica/train.txt", 'r') as seq_f:
        lines = seq_f.readlines()
        # 存放所有新商品id与新类别id的对应列表
        total_n2n_list = []
        no = 1
        # 记录已处理过的新商品id
        record_nid = []
        with open("niid_2_ncid.txt", 'a') as f:
            for line in lines:
                if no % 10000 == 0:
                    print("共{}行, 正在处理第{}行".format(len(lines), no))
                line_2_id_list = line.split(',')
                # 存放每一个[新商品id,新类别id]
                each_niid_2_ncid = []
                # 遍历每一个session,获取其对应的类别id列表
                for each_id in line_2_id_list:
                    # 如果该id已经处理过,则跳过
                    if each_id.strip() in record_nid:
                        continue
                    record_nid.append(each_id)
                    # nid -> oid
                    get_oid = nid2oid.get(each_id.strip())
                    # oid -> cate_id
                    get_category_id = cate_dict.get(get_oid)
                    # cate_id -> n_cate_id
                    get_n_category_id = oid2nid_category_dict.get(get_category_id)
                    # each_niid_2_ncid.append(each_id)
                    # each_niid_2_ncid.append(get_n_category_id)
                    content = str(each_id.strip()) + ',' + str(get_n_category_id)
                    f.write(content + '\n')
                    # total_n2n_list.append(each_niid_2_ncid)
                no += 1
            # for each_cate_list in total_n2n_list:
            #     # print(each_cate_list)
            #     write_category_seq(each_cate_list)


def item_price_seq():
    """
    根据商品序列获取对应的价格序列
    """
    # nid: oid
    nid2oid = {}

    # oid: price_id
    price_dict = {}

    # nid : price_id
    with open("../../datasets/products.csv", 'r') as cate_f:
        # oid : price_id
        # 获取类别对照表的所有行
        product_lines = cate_f.readlines()
        for each_line in product_lines:
            line_2_list = each_line.split(';')
            price_dict[line_2_list[0]] = line_2_list[1].strip()

    with open("../../new_oid2nid.csv", 'r') as o2nid_f:
        # nid : oid
        # 获取ID对照表的所有行
        oid2nid_lines = o2nid_f.readlines()
        for each_id_line in oid2nid_lines:
            id_line_2_list = each_id_line.split(',')
            # 根据新ID找到原始ID
            nid2oid[id_line_2_list[1].strip()] = id_line_2_list[0]

    with open("../../datasets/diginetica/train.txt", 'r') as seq_f:
        lines = seq_f.readlines()
        no = 1
        # 记录已处理过的新商品id
        record_nid = []
        with open("niid_2_priceid.txt", 'a') as f:
            for line in lines:
                if no % 10000 == 0:
                    print("共{}行, 正在处理第{}行".format(len(lines), no))
                line_2_id_list = line.split(',')
                # 遍历每一个session,获取其对应的类别id列表
                for each_id in line_2_id_list:
                    # 如果该id已经处理过,则跳过
                    if each_id.strip() in record_nid:
                        continue
                    record_nid.append(each_id)
                    # nid -> oid
                    get_oid = nid2oid.get(each_id.strip())
                    # oid -> price_id
                    get_price_id = price_dict.get(get_oid)
                    content = str(each_id.strip()) + ',' + str(get_price_id)
                    f.write(content + '\n')
                no += 1


if __name__ == '__main__':
    # itemid_2_categoryid()
    # del_line()
    # half_train_seq()

    # 根据session信息获取对应的item类别序列
    # itemid_2_categoryid_fast()

    # 将类别序列转成数据集格式
    # train_cate_seq_2_format()

    # 更新类别ID并重新保存成序列文件
    # update_category_id_2_train()

    # 获取新商品ID与新类别ID的映射文件
    new_itemID_2_new_categoryID()

    # 获取新商品ID与价格ID的映射文件
    item_price_seq()
