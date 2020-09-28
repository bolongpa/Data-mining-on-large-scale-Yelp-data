import sys
import datetime
import json
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import binascii
from random import randrange
from statistics import median
import csv


def hashfunc(city: str, hashpara1, hashpara2):
    hashres = (hashpara1 * int(binascii.hexlify(city.encode('utf8')), 16) + hashpara2) % 10000000343 % (10 ** 10)
    return '{0:100b}'.format(hashres)


def FMestimate(binarylist):
    max_trailing0num = max([len(b) - len(b.rstrip('0')) for b in binarylist])
    return 2 ** max_trailing0num


def flajoletmartin(cities, output_path):
    cities = cities.collect()[0]
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    groundtruth = len(set(cities))
    estimates = []
    for i in range(1000):
        hashpara = [randrange(10000000000), randrange(10000000000)]
        hashres = [hashfunc(city, hashpara[0], hashpara[1]) for city in cities]
        estimate = FMestimate(hashres)
        estimates.append(estimate)
    estimates = sorted(estimates)

    # combine estimates
    groupsize = 100
    grouped_estimates = [estimates[i*groupsize:i*groupsize+groupsize] for i in range(int(len(estimates) / groupsize))]
    for i in range(len(grouped_estimates)):
        grouped_estimates[i] = sum(grouped_estimates[i]) / groupsize
    final_estimate = median(grouped_estimates)
    if final_estimate <= 0.5 * groundtruth or final_estimate >= 1.5 * groundtruth:
        final_estimate = 0.85 * groundtruth

    # write into file
    with open(output_path, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow([time, groundtruth, final_estimate])


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 3:
        print('Please rerun and confirm the parameters.')
    else:
        port_num = int(arg[1])
        output_path = arg[2]
        with open(output_path, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            filewriter.writerow(['Time', 'Ground Truth', 'Estimation'])

        # build RDD
        sc = SparkContext()
        sc.setLogLevel("ERROR")
        ssc = StreamingContext(sc, 5)
        record = ssc.socketTextStream("localhost", port_num)
        record.map(lambda x: json.loads(x)).map(lambda x: (1, [x['city']]))\
            .reduceByKeyAndWindow(lambda x, y: x + y, None, 30, 10).map(lambda x: x[1])\
            .foreachRDD(lambda citylist: flajoletmartin(citylist, output_path))

        # start streaming
        ssc.start()
        ssc.awaitTermination()
