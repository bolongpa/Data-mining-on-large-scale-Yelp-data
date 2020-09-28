import sys
import glob
import random
import time
import csv
import json


def sq_mahalanobisDist(clstr_info, point, d):  # point without point index
    clstrN = clstr_info[0]
    clstrSUM = clstr_info[1]
    clstrSUMSQ = clstr_info[2]
    clstr_center = clstr_info[3]
    var = [clstrSUMSQ[i]/clstrN - (clstrSUM[i]/clstrN) ** 2 for i in range(d)]
    # compute squared distance
    temp = [(point[i] - clstr_center[i])**2/(var[i]+6.9533558078350043e-300) for i in range(d)]
    m_sq_dist = sum(temp)
    return m_sq_dist


def distance(p1, p2, d):  # squared Euclidean distance of p1, p2
    return sum([(p1[i] - p2[i]) ** 2 for i in range(d)])


def randcentroid(k, num_d):  # randomly pick k centroids as start
    inicentroids = []
    random_range = 10000  # each coordinate of generated centroid point is in [-random_range, random_range]
    for i in range(k):
        inicentroids.append([random.uniform(-random_range, random_range) for i in range(num_d)])  # a random centroid
    return inicentroids


def allols(data, centroids: list, d):  # how data points allocated to the centroids
    cenList = []
    for point in data:
        closest_centroid_index = 0  # initialization with the first centroid
        min_distance = distance(centroids[0], point, d)
        for i in range(1, len(centroids)):  # travel remaining centroids to get nearest one
            tmp_dist = distance(centroids[i], point, d)
            if tmp_dist < min_distance:
                min_distance = tmp_dist
                closest_centroid_index = i
        cenList.append(closest_centroid_index)
    return cenList  # [cen_num,] corresponding to the positions of points in the data set


def new_centroids(data, allocation, k, d, farthest_index):
    new_centroids = []
    temp = [0]*d
    for i in range(k):
        n = 0
        for p in range(len(allocation)):
            if allocation[p] == i:
                for j in range(d):
                    temp[j] += data[p][j]
                n += 1
        if n == 0:
            new_centroids.append(data[farthest_index])
        else:
            new_centroids.append([x / n for x in temp])
        temp = [0]*d
    return new_centroids


def kmeans(data, k, d):
    # find the farthest point from center of the entire dataset in case the number of clusters decrease
    datacenter = [sum(c) / len(data) for c in zip(*data)]  # compute the center of the entire data set
    farthest = data[0]  # initialize the farthest point with the first point
    for p in data:
        if distance(p, datacenter, d) > distance(farthest, datacenter, d):
            farthest = p
    farthest_point_index = data.index(farthest)

    # K-means
    centers = random.choices(data, k=k)  # randomly initialize centroids
    old_centers = []
    allo = []
    iteration_num = 0  # counter
    while centers != old_centers and iteration_num < 100:  # stop when convergence happen or reaching max ireration
        allo = allols(data, centers, d)
        old_centers = centers
        centers = new_centroids(data, allo, k, d, farthest_point_index)
        iteration_num += 1
    return allo, centers


def mergeCS(CS, d):
    new_CS = []
    merged_id = []
    for i in range(len(CS)):
        for j in range(i+1, len(CS)):
            if i not in merged_id and j not in merged_id:
                if sq_mahalanobisDist(CS[i], CS[j][3], d) < cluster_threshold or sq_mahalanobisDist(CS[j], CS[i][3], d) < cluster_threshold:
                    # merge the two CS
                    mergedCS = []
                    mergedCS.append(CS[i][0] + CS[j][0])  # N
                    mergedCS.append([CS[i][1][m]+CS[j][1][m] for m in range(d)])  # SUM
                    mergedCS.append([CS[i][2][m]+CS[j][2][m] for m in range(d)])  # SUMSQ
                    mergedCS.append([x / mergedCS[0] for x in mergedCS[1]])  # centroid of merged CS
                    mergedCS.append(CS[i][4] + CS[j][4])  # points
                    new_CS.append(mergedCS)
                    # mark i j as merged
                    merged_id.append(i)
                    merged_id.append(j)
    for n in range(len(CS)):  # put not merged old CS into new CS
        if n not in merged_id:
            new_CS.append(CS[n])
    return new_CS


if __name__ == '__main__':
    arg = sys.argv
    if len(arg) != 5:
        print('Please rerun and confirm the parameters.')
    else:
        start_time = time.time()
        input_folder_path = arg[1]
        n_cluster = int(arg[2])  # number of clusters
        out_cluster_res = arg[3]  # out_file1
        out_intermediate_res = arg[4]  # out_file2

        datafiles = glob.glob(input_folder_path + "/*.txt")  # all files in a list

        DS = []  # discard set  !![[N, SUM, SUMSQ, centroids, cluster_index], ]
        RS = []  # retained set  !!list of points in RS (with index)
        CS = []  # compression set  !![[N, SUM, SUMSQ, CScentroid, list of points in this CS], ]
        res = dict()  # points already allocated to DS  !! {point_index: DS_cen_id, }

        # K-means on first data file
        tempK = 5 * n_cluster  # large number of centroids for first K-means implementation

        with open(datafiles[0]) as f:
            total_data_w_id = [[float(n) for n in s[:-1].split(',')] for s in f.readlines()]
            if len(total_data_w_id) > 10000:
                sample_data = total_data_w_id[:10000]  # take first 10,000 data as sample
            else:
                sample_data = total_data_w_id
            dim = len(sample_data[0])-1  # dimension of the data points
            cluster_threshold = 3 * dim  # threshold to determine to put point into cluster
            allocation1, _ = kmeans([p[1:] for p in sample_data], tempK, dim)  # use allocation1 and sample_data we can know which point allocated to which centroid

            # get centroid indexes that contain within 100 points
            rare_centroids = []
            for i in range(tempK):
                if 0 < allocation1.count(i) <= 3:
                    rare_centroids.append(i)

            # move points allocated to rare centroids to RS (tempRS)
            outlier_index = []  # index of outlier points
            tempRS = []  # with first coordinate being the point index
            for i in range(len(allocation1)):
                if allocation1[i] in rare_centroids:
                    outlier_index.append(i)
                    tempRS.append(sample_data[i])
            sample_data_nonoutlier = [v for i, v in enumerate(sample_data) if i not in outlier_index]

            # rerun K-means on non-outliers and generate DS statistics
            allocation2, centroids2 = kmeans([p[1:] for p in sample_data_nonoutlier], n_cluster, dim)
            # put DS points to res
            for p in range(len(sample_data_nonoutlier)):
                res[int(sample_data_nonoutlier[p][0])] = allocation2[p]
            # generate DS stat
            for i in range(n_cluster):
                N = allocation2.count(i)
                points_in_cluster = [sample_data_nonoutlier[j][1:] for j in range(len(allocation2)) if allocation2[j] == i]  # all points in cluster i
                SUM = [sum(c) for c in zip(*points_in_cluster)]
                SUMSQ = [sum([k ** 2 for k in c]) for c in zip(*points_in_cluster)]
                DS.append([N, SUM, SUMSQ, centroids2[i], i])  # [[N, SUM, SUMSQ, centroids, cluster_index], ]


            # generate CS from RS and get new RS from outlier points
            if tempRS:  # if tempRS not empty
                allocation3, centroids3 = kmeans([p[1:] for p in tempRS], tempK, dim)
                rs_centroid_index = []  # centroid indexes containing only 1 point, so should be put into RS
                for i in range(tempK):  # index of clusters
                    if allocation3.count(i) == 1:
                        rs_centroid_index.append(i)
                # put clusters(points) with only 1 point to RS
                for i in range(len(allocation3)):
                    if allocation3[i] in rs_centroid_index:
                        RS.append(tempRS[i])  # real RS after initialization
                # put other clusters to CS
                CSdict = dict()  # {CenNum:[points],}
                for i in range(tempK):
                    if allocation3.count(i) > 1:
                        CSdict[i] = []
                for i in range(len(allocation3)):
                    if allocation3[i] in CSdict.keys():  # check if the point allocation should in CS
                        CSdict[allocation3[i]].append(tempRS[i])
                for k in CSdict.keys():
                    N = len(CSdict[k])
                    SUM = [sum(c) for c in zip(*CSdict[k])][1:]
                    SUMSQ = [sum([k ** 2 for k in c]) for c in zip(*CSdict[k])][1:]
                    CS.append([N, SUM, SUMSQ, centroids3[k], CSdict[k]])
        # complete initializing DS, CS, RS

        # process remaining points in first file
        remaining_points = total_data_w_id[10000:]
        # assign to DS
        for p in remaining_points:
            cluster_index_in_DS = 0
            p_c_sq_dist = sq_mahalanobisDist(DS[0], p[1:], dim)  # store the distance from the point to each cluster
            for i in range(1, n_cluster):
                if sq_mahalanobisDist(DS[i], p[1:], dim) < p_c_sq_dist:
                    p_c_sq_dist = sq_mahalanobisDist(DS[i], p[1:], dim)
                    cluster_index_in_DS = DS[i][4]
            if p_c_sq_dist < cluster_threshold:  # update DS if in one DS
                DS[cluster_index_in_DS][0] += 1  # update N
                DS[cluster_index_in_DS][1] = [DS[cluster_index_in_DS][1][i] + p[1:][i] for i in range(dim)]  # update SUM
                DS[cluster_index_in_DS][2] = [DS[cluster_index_in_DS][2][i] + (p[1:][i] ** 2) for i in range(dim)]  # update SUMSQ
                DS[cluster_index_in_DS][3] = [DS[cluster_index_in_DS][1][i] / DS[cluster_index_in_DS][0] for i in range(dim)]  # update centroid
                res[int(p[0])] = DS[cluster_index_in_DS][4]
            else:
                if CS:  # p cannot be assigned to DS, check in CS
                    cluster_index_in_CS = 0
                    p_c_sq_dist = sq_mahalanobisDist(CS[0], p[1:], dim)  # store the distance from the point to each cluster
                    for i in range(1, len(CS)):
                        if sq_mahalanobisDist(CS[i], p[1:], dim) < p_c_sq_dist:
                            p_c_sq_dist = sq_mahalanobisDist(CS[i], p[1:], dim)
                            cluster_index_in_CS = i
                    if p_c_sq_dist < cluster_threshold:  # update DS if in one DS
                        CS[cluster_index_in_CS][0] += 1  # update N
                        CS[cluster_index_in_CS][1] = [CS[cluster_index_in_CS][1][i] + p[1:][i] for i in
                                                      range(dim)]  # update SUM
                        CS[cluster_index_in_CS][2] = [CS[cluster_index_in_CS][2][i] + (p[1:][i] ** 2) for i in
                                                      range(dim)]  # update SUMSQ
                        CS[cluster_index_in_CS][3] = [CS[cluster_index_in_CS][1][i] / CS[cluster_index_in_CS][0] for i in
                                                      range(dim)]  # update centroid
                        CS[cluster_index_in_CS][4].append(p)
                    else:  # p cannot be assigned to CS, put in RS
                        RS.append(p)
                else:  # there is no CS
                    RS.append(p)
        # run K-means on RS if RS not empty
        if RS:
            allocation4, centroids4 = kmeans([p[1:] for p in RS], tempK, dim)
            tempRS = []
            rs_centroid_index = []  # centroid indexes that contain only 1 point, so should be put into RS
            for i in range(tempK):  # index of clusters
                if allocation4.count(i) == 1:
                    rs_centroid_index.append(i)
            # put clusters(points) with only 1 point to RS
            for i in range(len(allocation4)):
                if allocation4[i] in rs_centroid_index:
                    tempRS.append(RS[i])
            # put other clusters to CS
            CSdict = dict()  # {CenNum:[points],}
            for i in range(tempK):
                if allocation4.count(i) > 1:
                    CSdict[i] = []
            for i in range(len(allocation4)):
                if allocation4[i] in CSdict.keys():  # check if the point allocation should in CS
                    CSdict[allocation4[i]].append(RS[i])
            for k in CSdict.keys():  # summarize new CS and put into CS list
                N = len(CSdict[k])
                SUM = [sum(c) for c in zip(*CSdict[k])][1:]
                SUMSQ = [sum([k ** 2 for k in c]) for c in zip(*CSdict[k])][1:]
                CS.append([N, SUM, SUMSQ, centroids4[k], CSdict[k]])
            RS = tempRS
        # merge CS if two CS close enough
        if CS:
            CS = mergeCS(CS, dim)
        # output first intermediate result of first file
        with open(out_intermediate_res, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['round_id', 'nof_cluster_discard', 'nof_point_discard', 'nof_cluster_compression', 'nof_point_compression', 'nof_point_retained'])  # header
            writer.writerow([1, len(DS), sum([ds[0] for ds in DS]), len(CS), sum([cs[0] for cs in CS]), len(RS)])

        # BFR procedure to process remaining files
        for file_id in range(1, len(datafiles)):
            with open(datafiles[file_id]) as f:
                remaindata = [[float(n) for n in s[:-1].split(',')] for s in f.readlines()]
                for p in remaindata:
                    cluster_index_in_DS = 0
                    p_c_sq_dist = sq_mahalanobisDist(DS[0], p[1:],
                                                     dim)  # store the distance from the point to each cluster
                    for i in range(1, n_cluster):
                        if sq_mahalanobisDist(DS[i], p[1:], dim) < p_c_sq_dist:
                            p_c_sq_dist = sq_mahalanobisDist(DS[i], p[1:], dim)
                            cluster_index_in_DS = DS[i][4]
                    if p_c_sq_dist < cluster_threshold:  # update DS if in one DS
                        DS[cluster_index_in_DS][0] += 1  # update N
                        DS[cluster_index_in_DS][1] = [DS[cluster_index_in_DS][1][i] + p[1:][i] for i in
                                                      range(dim)]  # update SUM
                        DS[cluster_index_in_DS][2] = [DS[cluster_index_in_DS][2][i] + (p[1:][i] ** 2) for i in
                                                      range(dim)]  # update SUMSQ
                        DS[cluster_index_in_DS][3] = [DS[cluster_index_in_DS][1][i] / DS[cluster_index_in_DS][0] for i
                                                      in range(dim)]  # update centroid
                        res[int(p[0])] = DS[cluster_index_in_DS][4]
                    else:  # p cannot be assigned to DS, check in CS
                        if CS:
                            cluster_index_in_CS = 0
                            p_c_sq_dist = sq_mahalanobisDist(CS[0], p[1:],
                                                             dim)  # store the distance from the point to each cluster
                            for i in range(1, len(CS)):
                                if sq_mahalanobisDist(CS[i], p[1:], dim) < p_c_sq_dist:
                                    p_c_sq_dist = sq_mahalanobisDist(CS[i], p[1:], dim)
                                    cluster_index_in_CS = i
                            if p_c_sq_dist < cluster_threshold:  # update DS if in one DS
                                CS[cluster_index_in_CS][0] += 1  # update N
                                CS[cluster_index_in_CS][1] = [CS[cluster_index_in_CS][1][i] + p[1:][i] for i in
                                                              range(dim)]  # update SUM
                                CS[cluster_index_in_CS][2] = [CS[cluster_index_in_CS][2][i] + (p[1:][i] ** 2) for i in
                                                              range(dim)]  # update SUMSQ
                                CS[cluster_index_in_CS][3] = [CS[cluster_index_in_CS][1][i] / CS[cluster_index_in_CS][0] for
                                                              i in
                                                              range(dim)]  # update centroid
                                CS[cluster_index_in_CS][4].append(p)
                            else:  # p cannot be assigned to CS, put in RS
                                RS.append(p)
                        else:  # p cannot be assigned to CS, put in RS
                            RS.append(p)
                # run K-means on RS
                if RS:
                    allocation5, centroids5 = kmeans([p[1:] for p in RS], tempK, dim)
                    tempRS = []
                    rs_centroid_index = []  # centroid indexes containing only 1 point, so should be put into RS
                    for i in range(tempK):  # index of clusters
                        if allocation5.count(i) == 1:
                            rs_centroid_index.append(i)
                    # put clusters(points) with only 1 point to RS
                    for i in range(len(allocation5)):
                        if allocation5[i] in rs_centroid_index:
                            tempRS.append(RS[i])
                    # put other clusters to CS
                    CSdict = dict()  # {CenNum:[points],}
                    for i in range(tempK):
                        if allocation5.count(i) > 1:
                            CSdict[i] = []
                    for i in range(len(allocation5)):
                        if allocation5[i] in CSdict.keys():  # check if the point allocation should in CS
                            CSdict[allocation5[i]].append(RS[i])
                    for k in CSdict.keys():
                        N = len(CSdict[k])
                        SUM = [sum(c) for c in zip(*CSdict[k])][1:]
                        SUMSQ = [sum([k ** 2 for k in c]) for c in zip(*CSdict[k])][1:]
                        CS.append([N, SUM, SUMSQ, centroids5[k], CSdict[k]])
                    RS = tempRS
                # merge CS if two CS close enough
                if CS:
                    CS = mergeCS(CS, dim)
                # output intermediate result of this file
                with open(out_intermediate_res, 'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow([file_id+1, len(DS), sum([ds[0] for ds in DS]), len(CS), sum([cs[0] for cs in CS]), len(RS)])

        # merge CS to closest DS
        if CS:
            for CSc in CS:
                CScen = CSc[3]  # centroid of this CS cluster
                cluster_index_in_DS = 0
                c_c_sq_dist = sq_mahalanobisDist(DS[0], CScen, dim)  # store the distance from the centroid to each DS cluster
                for i in range(1, n_cluster):
                    if sq_mahalanobisDist(DS[i], CScen, dim) < c_c_sq_dist:
                        c_c_sq_dist = sq_mahalanobisDist(DS[i], CScen, dim)
                        cluster_index_in_DS = DS[i][4]
                for p in CSc[4]:  # put points in this CS into the closest DS
                    res[int(p[0])] = cluster_index_in_DS
        # put RS to closest DS
        if RS:
            for rsp in RS:
                tempdist = distance(rsp[1:], DS[0][3], dim)
                cluster_index_in_DS = 0
                for i in range(1, n_cluster):
                    if distance(rsp[1:], DS[i][3], dim) < tempdist:
                        tempdist = distance(rsp[1:], DS[i][3], dim)
                        cluster_index_in_DS = DS[i][4]
                res[int(rsp[0])] = cluster_index_in_DS

        # output to json file
        sorted_res = {str(i): res[i] for i in sorted(list(res.keys()))}
        with open(out_cluster_res, 'w') as f:
            json.dump(sorted_res, f)

        print("Duration:", (time.time() - start_time))
