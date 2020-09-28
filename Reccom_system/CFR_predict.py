import sys
from os import path
import time
import json
from pyspark import SparkContext


def item_predicting_func(target_usr, target_business, usrrated_items, itemmodel, b_avg):
    if target_usr in usrrated_items.keys():
        rated_item_star = usrrated_items[target_usr]  # (businesses, star) rated by this user
        # check rated_item_star length and sort, take at most 5
        temp = dict()
        for i in rated_item_star:
            if frozenset((i[0], target_business)) in itemmodel.keys():
                temp[i] = itemmodel[frozenset((i[0], target_business))]
        if len(temp) > 5:
            temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:5]  # sorting and take first 5
            rated_item_star = [j[0] for j in temp]

        n = 0  # prediction numerator
        d = 0  # prediction denominator
        for b_s in rated_item_star:
            if frozenset((b_s[0], target_business)) in itemmodel.keys():
                n += b_s[1] * itemmodel[frozenset((b_s[0], target_business))]
                d += abs(itemmodel[frozenset((b_s[0], target_business))])
        if d != 0:  #
            prediction = n / d
            return [target_usr, target_business, prediction]
        else:
            if target_business in b_avg.keys():
                return [target_usr, target_business, b_avg[target_business]]
    else:
        if target_business in b_avg.keys():
            return [target_usr, target_business, b_avg[target_business]]


def item_based_predict(train_file, test_file, model_file, output_path):
    # read avg
    path_head = path.split(train_file)[0]
    if path_head == '':
        b_avg_path = 'business_avg.json'
    else:
        b_avg_path = path.split(train_file)[0] + '/business_avg.json'
    with open(b_avg_path, 'r') as avg:
        b_avg = json.loads(avg.read())

    # build RDD
    sc = SparkContext()
    item_model = sc.textFile(model_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (frozenset((x['b1'], x['b2'])), x['sim'])).collectAsMap()
    usr_b_and_s = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], [(x['business_id'], x['stars'])])) \
        .reduceByKey(lambda x, y: x + y).collectAsMap()  # {usr:[(b_id, star),],}

    targetpredictrdd = sc.textFile(test_file).map(lambda x: json.loads(x)) \
        .map(lambda x: item_predicting_func(x['user_id'], x['business_id'], usr_b_and_s, item_model, b_avg)) \
        .filter(lambda x: x is not None).map(lambda x: {'user_id': x[0], 'business_id': x[1], 'stars': x[2]}).collect()
    # print(targetpredictrdd)

    # write to output
    with open(output_path, 'w') as o:
        for i in targetpredictrdd:
            o.write(json.dumps(i) + '\n')


def user_predicting_func(target_usr, target_business, item_ratedusers, usermodel, u_avg, u_ratenum):
    if target_usr in u_avg.keys():
        r_targetu_avg = u_avg[target_usr]
        n = 0  # prediction numerator
        d = 0  # prediction denominator
        if target_business in item_ratedusers.keys():
            rated_usr_star = item_ratedusers[target_business]  # users rated the target_business {usr: star,}
            # check rated_usr_star length and sort, take at most 5
            temp = dict()
            for other_u in rated_usr_star.keys():  # other_u is usr_id
                if frozenset((target_usr, other_u)) in usermodel.keys():
                    temp[other_u] = usermodel[frozenset((target_usr, other_u))]  #{u_id:sim,}
            if len(temp) > 15:
                temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)[:15]  # sorting and take first 5, [(u_id, sim),]
                rated_usr_star = {k[0]:rated_usr_star[k[0]] for k in temp}
            
            for other_u in rated_usr_star.keys():
                if frozenset((target_usr, other_u)) in usermodel.keys():
                    d += abs(usermodel[frozenset((target_usr, other_u))])
                    # compute numerator factor
                    n += (item_ratedusers[target_business][other_u] - (u_avg[other_u] * u_ratenum[other_u] - item_ratedusers[target_business][other_u]) / (u_ratenum[other_u] - 1)) * usermodel[frozenset((target_usr, other_u))]
                    if d != 0:
                        prediction = min(r_targetu_avg + n / d, 5)
                        return [target_usr, target_business, prediction]
                else:
                    prediction = r_targetu_avg
                    return [target_usr, target_business, prediction]
        else:
            prediction = r_targetu_avg
            return [target_usr, target_business, prediction]


def user_based_predict(train_file, test_file, model_file, output_path):
    # read avg
    path_head = path.split(train_file)[0]
    if path_head == '':
        u_avg_path = 'user_avg.json'
    else:
        u_avg_path = path.split(train_file)[0] + '/user_avg.json'
    with open(u_avg_path, 'r') as avg:
        u_avg = json.loads(avg.read())

    # build RDD
    sc = SparkContext()
    user_model = sc.textFile(model_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (frozenset((x['u1'], x['u2'])), x['sim'])).collectAsMap()
    train_rdd = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id'], x['stars'])).cache()
    business_usr_and_s = train_rdd.map(lambda x: (x[1], [(x[0], x[2])])) \
        .reduceByKey(lambda x, y: x + y)\
        .mapValues(lambda x: {i[0]:i[1] for i in x}).collectAsMap()  # {b_id:{usr: star,},}
    user_ratenum = train_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()  # {usr: ratenum,}

    targetpredictrdd = sc.textFile(test_file).map(lambda x: json.loads(x)) \
        .map(lambda x: (x['user_id'], x['business_id']))\
        .map(lambda x: user_predicting_func(x[0], x[1], business_usr_and_s, user_model, u_avg,
                                            user_ratenum)) \
        .filter(lambda x: x is not None).map(lambda x: {'user_id': x[0], 'business_id': x[1], 'stars': x[2]}).collect()
    # print(targetpredictrdd)

    # write to output
    with open(output_path, 'w') as o:
        for i in targetpredictrdd:
            o.write(json.dumps(i) + '\n')


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 6:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        train_file_path = arg[1]  # train_file
        test_file_path = arg[2]  # test_file
        model_file_path = arg[3]  # model_file
        output_file_path = arg[4]  # output_file
        cf_type = arg[5]  # item_based or user_based

        if cf_type == 'item_based':
            item_based_predict(train_file_path, test_file_path, model_file_path, output_file_path)
        if cf_type == 'user_based':
            user_based_predict(train_file_path, test_file_path, model_file_path, output_file_path)

        print("Duration:", (time.time() - starttime))
