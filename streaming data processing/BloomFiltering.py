import sys
import time
import json
from pyspark import SparkContext
import binascii


def genebitarrary_hash(cityies: str, m, hashpara):
    bitarray = []
    for city in cityies:
        for para in hashpara:
            hashres = (para[0] * int(binascii.hexlify(city.encode('utf8')), 16) + para[1]) % 10039 % m
            if hashres not in bitarray:
                bitarray.append(hashres)
    return bitarray


def hashfunc(cityname: str, m, hashpara):
    res = []
    for para in hashpara:
        hashres = (para[0] * int(binascii.hexlify(cityname.encode('utf8')), 16) + para[1]) % 10039 % m
        res.append(hashres)
    return res


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 4:
        print('Please rerun and confirm the parameters.')
    else:
        starttime = time.time()
        first_path = arg[1]
        second_path = arg[2]
        output_path = arg[3]

        # build RDD
        sc = SparkContext()
        data = sc.textFile(first_path).map(lambda x: json.loads(x)) \
            .map(lambda x: x['city']).distinct().filter(lambda x: x != '').collect()

        # hash parameters
        hashpara = [[18124, 99463], [71730, 39428], [13483, 99262], [90482, 9985], [13453, 90097], [25413, 40901], [69022, 50309], [6275, 68046], [27354, 50241], [14428, 51089]]
        m = 10000  # num of buckets
        bitarray = genebitarrary_hash(data, m, hashpara)

        # prediction
        output = []
        pre_data = sc.textFile(second_path).map(lambda x: json.loads(x)) \
            .map(lambda x: x['city']).collect()
        for p_city in pre_data:
            if p_city == '':
                output.append(0)
            else:
                predictionres = 1
                for p in hashfunc(p_city, m, hashpara):
                    if p not in bitarray:
                        predictionres = 0
                        break
                output.append(predictionres)

        with open(output_path, 'w') as f:
            for i in output:
                f.write(str(i) + ' ')

        print("Duration:", (time.time() - starttime))
