import sys
from operator import add
from pyspark import SparkContext
import time


def checkExistinBasket(itemset, basket):  # check if this basket contributes to support to this itemset
    exist = True
    for i in itemset:
        if i not in basket:
            exist = False
    return exist


def APriori_1(sample, ps):
    sample = list(sample)
    itemsets = dict()  # all singletons in this sample
    frequentItemsets = []  # all frequent singletons in this sample
    for b in sample:  # travel each basket in sample
        for i in b:  # travel each item in this basket
            if i not in itemsets:
                itemsets[i] = 1
                if itemsets[i] >= ps:
                    frequentItemsets.append(i)
                    yield tuple([i])
            else:
                if itemsets[i] < ps:
                    itemsets[i] += 1
                    if itemsets[i] >= ps:
                        frequentItemsets.append(i)
                        yield tuple([i])


def APriori_2(sample, singleton_list, ps):
    # get frequent 2-itemsets in this sample (because too many pairs to check)
    itemsets = dict()  # all possible pairs
    for b in sample:
        for i in range(len(b) - 1):
            for j in range(i + 1, len(b)):
                if b[i] in singleton_list and b[j] in singleton_list:
                    pair = frozenset([b[i], b[j]])
                    if pair not in itemsets.keys():
                        itemsets[pair] = 1
                    else:
                        itemsets[pair] += 1
    frequentItemsets = []
    for p in itemsets.keys():
        if itemsets[p] >= ps:
            frequentItemsets.append(tuple(p))
            yield tuple(p)


def APriori_more(sample, pairs, ps):
    item_num = 2
    frequentItemsets = pairs
    # generate itemsets larger than 2
    while frequentItemsets != []:
        item_num += 1
        tempitemsets_dict = dict()
        combination = []  # frozensets in list
        # collect all possible itemsets with current item_num
        for i in range(len(frequentItemsets)-1):
            for j in range(i+1, len(frequentItemsets)):
                temp = set(frequentItemsets[i]).union(set(frequentItemsets[j]))
                if len(temp) == item_num:
                    combination.append(frozenset(temp))
        for b in sample:
            for c in combination:
                if c.issubset(tuple(b)):
                    if c not in tempitemsets_dict.keys():
                        tempitemsets_dict[c] = 1
                    else:
                        tempitemsets_dict[c] += 1
        fruentsets = []
        for s in tempitemsets_dict.keys():
            if tempitemsets_dict[s] >= ps:
                fruentsets.append(tuple(s))
                yield tuple(s)
        frequentItemsets = fruentsets


def cnt_support(candi_Set, file):
    file = list(file)
    for b in file:
        for i in candi_Set:
            if len(i) == 1:
                if i[0] in b:
                    yield (i, 1)
            else:
                if set(i).issubset(b):
                    yield (i, 1)


def format_output(tag, out_list):
    out = str(tag) + ':\n'
    line_dic = dict()
    for i in out_list:
        length = len(i)
        if length not in line_dic.keys():
            line_dic[length] = []
            if length == 1:
                line_dic[length].append(i[0])
            elif length > 1:
                line_dic[length].append(sorted(i))
        else:
            if length == 1:
                line_dic[length].append(i[0])
            elif length > 1:
                line_dic[length].append(sorted(i))
    len_cnt = 1
    while len_cnt in line_dic.keys():
        for i in sorted(line_dic[len_cnt]):
            if len_cnt == 1:
                out += "('" + str(i) + "'),"
            else:
                out += str(tuple(i)) + ","
        out = out[:-1]  # remove last comma of the line
        out += '\n'
        out += '\n'  # blank line
        len_cnt += 1
    return out


if __name__ == "__main__":
    arg = sys.argv
    if len(arg) != 5:
        print('Please check the parameters and run again.')
    else:
        start_time = time.time()
        filter_threshold = int(arg[1])
        support = int(arg[2])
        input_file_path = arg[3]
        output_file_path = arg[4]

        # build RDD
        sc = SparkContext()
        rdd = sc.textFile(input_file_path)
        header = rdd.take(1)
        data = rdd.filter(lambda line: line != header[0])
        basketrdd = data.map(lambda x: [i for i in x.split(',')])\
            .map(lambda x: (x[0], [str(x[1])]))\
            .reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], list(set(x[1]))))\
            .filter(lambda x: len(x[1]) > filter_threshold).values().cache()


        ps = support / basketrdd.getNumPartitions()
        # generate singleton candidates
        son_candi_1 = basketrdd.mapPartitions(lambda x: APriori_1(x, ps)).distinct().collect()
        singletons = [x[0] for x in son_candi_1]
        son_candi = son_candi_1
        # generate pair candidates
        if singletons:
            son_candi_2 = basketrdd.mapPartitions(lambda x: APriori_2(x, singletons, ps)).distinct().collect()
            if son_candi_2:
                son_candi += son_candi_2
                # generate candidates size over 3
                son_candi_more = basketrdd.mapPartitions(lambda x: APriori_more(x, son_candi_2, ps)).distinct().collect()
                if son_candi_more:
                    son_candi += son_candi_more
        candi_out = format_output('Candidates', son_candi)

        # generate frequent itemsets
        son_frequent = basketrdd.mapPartitions(lambda x: cnt_support(son_candi, x)) \
            .reduceByKey(add) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: x[0]).collect()
        frequent_out = format_output('Frequent Itemsets', son_frequent)

        final_out = candi_out + frequent_out
        with open(output_file_path, 'w') as f:
            f.write(final_out)
        print("Duration:", (time.time() - start_time))  # print execution time in terminal
