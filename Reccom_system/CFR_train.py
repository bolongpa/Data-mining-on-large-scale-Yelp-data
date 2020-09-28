import sys
import time
import json
from math import sqrt
from pyspark import SparkContext
from itertools import combinations


def item_basedcf(train_file, model_file):
    # build RDD
    sc = SparkContext()
    rdd = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id'], int(x['stars']))).cache()
    business_user_star = rdd.map(lambda x: (x[1], [(x[0], x[2])])).reduceByKey(lambda x, y: x + y)\
        .mapValues(lambda l: {x[0]: x[1] for x in l}).collectAsMap()  # {b_id:{usr:star}}

    # get all possible business pairs with at least 3 co-rated users
    b_pair = rdd.map(lambda x: (x[0], {x[1]})).reduceByKey(lambda x, y: x | y)\
        .mapValues(lambda s: list(combinations(s, 2))).flatMap(lambda x: [(x[0], p) for p in x[1]])\
        .map(lambda x: (tuple(sorted(x[1])), {x[0]}))\
        .reduceByKey(lambda x, y: x | y).mapValues(lambda x: list(set(x)))\
        .filter(lambda x: len(x[1]) >= 3).collect()  # [(b_id_pair, [common usr,]),]

    # compute Pearson similarities of each pair and remove negative ones
    pearson_res = []
    for p_usr in b_pair:
        bi = p_usr[0][0]
        bj = p_usr[0][1]
        # compute average
        bi_avg = sum([business_user_star[bi][u] for u in p_usr[1]]) / len(p_usr[1])
        bj_avg = sum([business_user_star[bj][u] for u in p_usr[1]]) / len(p_usr[1])
        pearson_n = sum([(business_user_star[bi][u] - bi_avg) * (business_user_star[bj][u] - bj_avg) for u in p_usr[1]])
        if pearson_n > 0:  # only keep pairs with positive pearson correlation
            pearson_d = sqrt(sum([(business_user_star[bi][u] - bi_avg) ** 2 for u in p_usr[1]])) * sqrt(sum([(business_user_star[bj][u] - bj_avg) ** 2 for u in p_usr[1]]))
            if pearson_d != 0:
                pearson_cor = pearson_n / pearson_d
                pearson_res.append({'b1': bi, 'b2': bj, 'sim': pearson_cor})

    # output model: 557680 pairs
    with open(model_file, 'w') as m:
        for i in pearson_res:
            m.write(json.dumps(i) + '\n')


def minhash(rowindexes, rownum, hashnum):
    signatures = []  # to store hash signatures
    hashfunc1 = lambda x: (3 * x + 2) % rownum
    hashfunc2 = lambda x: (5 * x + 3) % rownum
    for i in range(1, hashnum + 1):
        hashfunc = lambda x: ((i * hashfunc1(x) + i * hashfunc2(x) + i * i) % 33569) % rownum
        # find signature using current hash function
        rowindexes_hash = [hashfunc(x) for x in list(rowindexes)]
        signatures.append(min(rowindexes_hash))
    return signatures


def breaktobands(signature: list, b: int, r: int):
    return [((i, tuple(signature[1][i * r:i * r + r])), [signature[0]]) for i in range(b)]


def user_basedcf(train_file, model_file):
    # obtain pairs
    # build RDD
    sc = SparkContext()
    rdd = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id']) if x['stars'] is not None else None) \
        .distinct()
    businessid = rdd.map(lambda x: x[1]).distinct().collect()
    num_row = len(businessid)  # number of distinct businesses
    rdd_prepare = rdd.map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .mapValues(lambda usrs: [businessid.index(u) for u in usrs])  # key: user_id; value: [row num of reviewed businesses]
    dict_usr = rdd_prepare.collectAsMap()
    # minhash
    num_hash = 23  # number of hash functions
    rdd_minhash = rdd_prepare.mapValues(lambda x: minhash(x, num_row, num_hash))  # (business_id, signature:list)
    # LSH
    b = 23  # number of bands
    r = int(num_hash / b)
    candidate_pairs = rdd_minhash.repartition(150).flatMap(lambda x: breaktobands(x, b, r)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: tuple(set(x[1]))).filter(lambda x: len(x) >= 2) \
        .flatMap(lambda x: list(combinations(x, 2))) \
        .distinct().collect()
    # generate result from candidates: 801432 pairs
    u_pair = dict()
    for p in candidate_pairs:  # p:tuple
        c1 = set(dict_usr[p[0]])  # set of row_num of usrs
        c2 = set(dict_usr[p[1]])
        similarity = len(c1.intersection(c2)) / len(c1.union(c2))
        if similarity >= 0.01 and len(c1.intersection(c2)) >= 3:
            u_pair[frozenset(p)] = list(c1.intersection(c2))  # {pair:[common_businessid,],}
    #print(len(u_pair.items()))

    # compute Pearson correlation and filter pairs
    user_business_star = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id'], int(x['stars']))).map(lambda x: (x[0], [(x[1], x[2])]))\
        .reduceByKey(lambda x, y: x + y).mapValues(lambda l: {x[0]: x[1] for x in l})\
        .collectAsMap()  # {usr:{b_id:star}}
    # compute Pearson similarities of each pair and remove negative ones
    pearson_res = []
    for usr_pair in list(u_pair.keys()):
        common_business = u_pair[usr_pair]  # :list
        ui = list(usr_pair)[0]
        uj = list(usr_pair)[1]
        # compute average
        ui_avg = sum([user_business_star[ui][businessid[b]] for b in common_business]) / len(common_business)
        uj_avg = sum([user_business_star[uj][businessid[b]] for b in common_business]) / len(common_business)
        pearson_n = sum([(user_business_star[ui][businessid[b]] - ui_avg) * (user_business_star[uj][businessid[b]] - uj_avg) for b in common_business])
        if pearson_n > 0:  # only keep pairs with positive pearson correlation
            pearson_d = sqrt(sum([(user_business_star[ui][businessid[b]] - ui_avg) ** 2 for b in common_business])) * sqrt(
                sum([(user_business_star[uj][businessid[b]] - uj_avg) ** 2 for b in common_business]))
            if pearson_d != 0:
                pearson_cor = pearson_n / pearson_d
                pearson_res.append({'u1': ui, 'u2': uj, 'sim': pearson_cor})
    #print(len(pearson_res))

    # output model: 399186 pairs
    with open(model_file, 'w') as m:
        for i in pearson_res:
            m.write(json.dumps(i) + '\n')



if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 4:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        train_file_path = arg[1]  # train_file
        model_file_path = arg[2]  # model_file
        cf_type = arg[3]  # item_based or user_based

        if cf_type == 'item_based':
            item_basedcf(train_file_path, model_file_path)
        if cf_type == 'user_based':
            user_basedcf(train_file_path, model_file_path)

        print("Duration:", (time.time() - starttime))
