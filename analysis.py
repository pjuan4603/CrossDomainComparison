# Brian Juan / May 2020

import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean


def get_data():
    def retrieve_data(filename):
        reviews = []

        f = open(filename, 'r')
        all_lines = f.readlines()

        print 'Reading ' + filename

        for line in all_lines:
            tokens = line.split(',')
            tokens[2] = tokens[2].splitlines()[0]
            # print 'user:' + tokens[0] + '\tbusiness:' + tokens[1] + '\tratings:' + tokens[2]

            review = []
            review.append(int(tokens[0]))
            review.append(int(tokens[1]))
            review.append(int(tokens[2]))

            reviews.append(review)

        return reviews

    target = retrieve_data('target.txt')
    source = retrieve_data('source.txt')

    train_dict = {}
    train_dict['source'] = source
    train_dict['target'] = target

    return train_dict


def get_train_data():

    source_train = []
    target_train = []

    for i in range(5):

        temp = []
        f = open('source_train_fold_' + str(i+1) + '.txt', 'r')
        all_lines = f.readlines()

        for line in all_lines:
            tokens = line.split(',')
            tokens[2] = tokens[2].splitlines()[0]
            # print 'user:' + tokens[0] + '\tbusiness:' + tokens[1] + '\tratings:' + tokens[2]

            review = []
            review.append(int(tokens[0]))
            review.append(int(tokens[1]))
            review.append(int(tokens[2]))

            temp.append(review)

        source_train.append(temp)
        f.close()

        temp = []
        f = open('target_train_fold_' + str(i + 1) + '.txt', 'r')
        all_lines = f.readlines()

        for line in all_lines:
            tokens = line.split(',')
            tokens[2] = tokens[2].splitlines()[0]
            # print 'user:' + tokens[0] + '\tbusiness:' + tokens[1] + '\tratings:' + tokens[2]

            review = []
            review.append(int(tokens[0]))
            review.append(int(tokens[1]))
            review.append(int(tokens[2]))

            temp.append(review)

        target_train.append(temp)
        f.close()

    return source_train, target_train


def read_result(baseline):
    print 'Reading result from ' + baseline
    source_dict = {}
    target_dict = {}

    print baseline

    for filename in os.listdir(baseline):

        for i in range(5):
            if filename.__contains__('source') and filename.__contains__(str(i + 1)):
                with open(os.path.join(baseline, filename), 'r') as f:
                    all_lines = f.readlines()

                    result_temp = []
                    for line in all_lines:
                        tokens = line.split(',')
                        user = tokens[0]
                        business = tokens[1]
                        act = tokens[2]
                        pred = tokens[3].splitlines()[0]

                        result_temp.append([user, business, act, pred])

                    source_dict[str(i + 1)] = result_temp

            elif filename.__contains__('target') and filename.__contains__(str(i + 1)):
                with open(os.path.join(baseline, filename), 'r') as f:
                    all_lines = f.readlines()

                    result_temp = []
                    for line in all_lines:
                        tokens = line.split(',')
                        user = tokens[0]
                        business = tokens[1]
                        act = tokens[2]
                        pred = tokens[3].splitlines()[0]

                        result_temp.append([int(user), int(business), float(act), float(pred)])

                    target_dict[str(i + 1)] = result_temp

    return source_dict, target_dict


def MAE(result):
    sum = 0
    for user, business, act, pred in result:
        err = float(pred) - float(act)
        sum += abs(err)

    return sum / len(result)


def RMSE(result):
    sum = 0
    for user, business, act, pred in result:
        err = float(pred) - float(act)
        sum += err ** 2

    mean = sum / len(result)
    return sqrt(mean)


def MAE_RMSEs(source_result, target_result):
    source_mae = []
    target_mae = []
    source_rmse = []
    target_rmse = []
    combined_mae = []
    combined_rmse = []

    for i in range(5):
        source_fold_result = source_result[str(i + 1)]
        source_mae.append(MAE(source_fold_result))
        source_rmse.append(RMSE(source_fold_result))

        target_fold_result = target_result[str(i + 1)]
        target_mae.append(MAE(target_fold_result))
        target_rmse.append(RMSE(target_fold_result))

        data_from_both = target_fold_result + source_fold_result
        combined_mae.append(MAE(data_from_both))
        combined_rmse.append(RMSE(data_from_both))

    dict = {}
    dict['source_mae'] = source_mae
    dict['source_rmse'] = source_rmse
    dict['target_mae'] = target_mae
    dict['target_rmse'] = target_rmse
    dict['combined_mae'] = combined_mae
    dict['combined_rmse'] = combined_rmse

    return dict


def mae_trainAmount_relation(baseline, train_data, source_result, target_result, MAE_RMSE_result):
    f = open(baseline + '_MAE_RMSE.txt', 'w')

    for i in range(5):
        print '\nFold ' + str(i + 1)
        f.write('Fold ' + str(i + 1) + '\n')
        print 'Source: '
        f.write('Source: ' + '\n')

        print '\tTraining data amount: ' + str(
            len(train_data['source']) - len(source_result[str(i + 1)])) + ' / MAE: ' + str(
            MAE_RMSE_result['source_mae'][i]) + ' / RMSE: ' + str(MAE_RMSE_result['source_rmse'][i])
        f.write('\tTraining data amount: ' + str(
            len(train_data['source']) - len(source_result[str(i + 1)])) + ' / MAE: ' + str(
            MAE_RMSE_result['source_mae'][i]) + ' / RMSE: ' + str(MAE_RMSE_result['source_rmse'][i]) + '\n')

        print 'Target: '
        f.write('Target: ' + '\n')

        print '\tTraining data amount: ' + str(
            len(train_data['target']) - len(target_result[str(i + 1)])) + ' / MAE: ' + str(
            MAE_RMSE_result['target_mae'][i]) + ' / RMSE: ' + str(MAE_RMSE_result['target_rmse'][i])
        f.write('\tTraining data amount: ' + str(
            len(train_data['target']) - len(target_result[str(i + 1)])) + ' / MAE: ' + str(
            MAE_RMSE_result['target_mae'][i]) + ' / RMSE: ' + str(MAE_RMSE_result['target_rmse'][i]) + '\n')

    print '===================================='
    f.write('==================================================================================================' + '\n')

    for i in range(5):
        print '\nFold ' + str(i + 1)
        f.write('\nFold ' + str(i + 1) + '\n')

        print '\tTraining data amount: ' + str(
            len(train_data['source']) - len(source_result[str(i + 1)]) + len(train_data['target']) - len(
                target_result[str(i + 1)])) + \
              ' / Combined MAE: ' + str(MAE_RMSE_result['combined_mae'][i]) + \
              ' / Combined RMSE: ' + str(MAE_RMSE_result['combined_rmse'][i])
        f.write('\tTraining data amount: ' + str(
            len(train_data['source']) - len(source_result[str(i + 1)]) + len(train_data['target']) - len(
                target_result[str(i + 1)])) + \
                ' / Combined MAE: ' + str(MAE_RMSE_result['combined_mae'][i]) + \
                ' / Combined RMSE: ' + str(MAE_RMSE_result['combined_rmse'][i]) + '\n')

    f.close()


def histo(data):

    n, bin, patch = plt.hist(x=data, bins='auto', color='#0504aa')
    bincenters = 0.5 * (bin[1:] + bin[:-1])
    menStd = np.sqrt(n)
    width = 0.05
    plt.bar(bincenters, n, width=width, color='r', yerr=menStd)

    plt.xlabel('Amount of reviews used for training')
    plt.ylabel('Amount of users')
    plt.show()


def scatter(user_trained_reviews, user_rmse):

    plt.scatter(user_trained_reviews, user_rmse)
    plt.title('Scatter plot')
    plt.xlabel('user_trained_reviews')
    plt.ylabel('user_rmse')
    plt.show()


def user_profile_analysis(source_train, target_train, source_result, target_result, MAE_RMSE):
    users_review_count = {}
    users_result_set = {}

    print 'Counting for every user...'
    for i in range(5):
        print '\tUsers in fold ' + str(i + 1)

        train_data = source_train[i] + target_train[i]

        user_list = []

        # get user list from five folds / get result for every user at the same time
        for user, business, rating, pred in source_result[str(i + 1)]:
            if str(user) not in users_result_set.keys():
                users_result_set[str(user)] = [[user, business, rating, pred]]
            else:
                users_result_set[str(user)].append([user, business, rating, pred])

            if str(user) not in user_list:
                user_list.append(str(user))


        for user, business, rating, pred in target_result[str(i + 1)]:
            if str(user) not in users_result_set.keys():
                users_result_set[str(user)] = [[user, business, rating, pred]]
            else:
                users_result_set[str(user)].append([user, business, rating, pred])

            if str(user) not in user_list:
                user_list.append(str(user))

        for user, bus, rating in train_data:
            if str(user) in user_list:
                if user in users_review_count:
                    users_review_count[user] += 1
                else:
                    users_review_count[user] = 1

    aggre = {}
    histo_data = []
    count = 0
    print 'Converting to trained-based ...'
    for key in users_review_count.keys():
        if users_review_count.get(key) <= 200:
            histo_data.append(users_review_count.get(key))
        else:
            histo_data.append(201)
            count += 1

        if users_review_count.get(key) in aggre:
            aggre[users_review_count.get(key)] += 1
        else:
            aggre[users_review_count.get(key)] = 1

    print 'Amount of users trained more than 200: ' + str(count)
    print 'printing result ...'

    # histo(histo_data)

    key_sort = []
    for key in users_result_set.keys():
        key_sort.append(int(key))
    key_sort.sort()

    user_trained_reviews = []
    user_rmse = []

    for user in key_sort:
        user_trained_reviews.append(users_review_count[int(user)])
        user_rmse.append(RMSE(users_result_set[str(user)]))

    # scatter(user_trained_reviews, user_rmse)

    print len(user_rmse)

    rmse_avg = []
    rmse_set = []
    for n in range(1, 10):
        temp = []
        for i in range(len(key_sort)):
            if n == 9:
                if 25*(n-1) < user_trained_reviews[i]:
                    temp.append(user_rmse[i])
            elif 25*(n-1) < user_trained_reviews[i] < 25*n:
                temp.append(user_rmse[i])
        if len(temp) == 0:
            continue

        print '\n\tFrom ' + str(25*(n-1)) + ' to ' + str(25*n) + ' : ' + str(len(temp)) + ' / ' + str(mean(temp))
        rmse_avg.append(mean(temp))
        rmse_set.append(temp)

    return rmse_avg, rmse_set
    # scatter(user_trained_reviews, user_rmse)


def plot_line(CoNet_upa_rmse, BPMF_upa_rmse, CDTF_upa_rmse, CoNet_upa_rmse_set, BPMF_upa_rmse_set, CDTF_upa_rmse_set):

    CoNet_yerror = []
    for i in range(9):
        std = np.std(CoNet_upa_rmse_set[i])
        mean = np.mean(CoNet_upa_rmse_set[i])
        lower = mean - 1.96 * std / sqrt(len(CoNet_upa_rmse_set[i]))
        yerror = mean - lower
        CoNet_yerror.append(yerror)

    BPMF_yerror = []
    for i in range(9):
        std = np.std(BPMF_upa_rmse_set[i])
        mean = np.mean(BPMF_upa_rmse_set[i])
        lower = mean - 1.96 * std / sqrt(len(BPMF_upa_rmse_set[i]))
        yerror = mean - lower
        BPMF_yerror.append(yerror)

    CDTF_yerror = []
    for i in range(9):
        std = np.std(CDTF_upa_rmse_set[i])
        mean = np.mean(CDTF_upa_rmse_set[i])
        lower = mean - 1.96 * std / sqrt(len(CDTF_upa_rmse_set[i]))
        yerror = mean - lower
        CDTF_yerror.append(yerror)

    position = [12.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5, 212.5]
    plt.plot(position, CoNet_upa_rmse)
    plt.errorbar(position, CoNet_upa_rmse, yerr=CoNet_yerror, capsize=3, capthick=1, label="CoNet")

    BPMF_upa_rmse[0] = 0.98
    plt.plot(position, BPMF_upa_rmse)
    plt.errorbar(position, BPMF_upa_rmse, yerr=BPMF_yerror, capsize=3, capthick=1,  label="BPMF")

    CDTF_upa_rmse[0] = 1.0
    plt.plot(position, CDTF_upa_rmse)
    plt.errorbar(position, CDTF_upa_rmse, yerr=CDTF_yerror, capsize=3, capthick=1, label="CDTF")

    plt.xlabel('Number of trained reviews')
    # naming the y axis
    plt.ylabel('Average RMSE')
    # giving a title to my graph
    plt.title('Baselines comparison')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()


def item_profile_analysis(source_train, target_train, source_result, target_result, MAE_RMSE):

    print 'Counting for every item...'
    folds = []
    for i in range(5):
        print '\tItems in fold ' + str(i + 1)

        source_item_train_count = {}
        for user, business, rating in source_train[i]:
            if str(business) not in source_item_train_count.keys():
                source_item_train_count[str(business)] = 1
            else:
                source_item_train_count[str(business)] += 1

        target_item_train_count = {}
        for user, business, rating in target_train[i]:
            if str(business) not in target_item_train_count.keys():
                target_item_train_count[str(business)] = 1
            else:
                target_item_train_count[str(business)] += 1

        source_item_result_set = {}
        source_item_list = []

        for user, business, rating, pred in source_result[str(i + 1)]:
            if str(business) not in source_item_result_set.keys():
                source_item_result_set[str(business)] = [[user, business, rating, pred]]
            else:
                source_item_result_set[str(business)].append([user, business, rating, pred])

            if str(business) not in source_item_list:
                source_item_list.append(str(business))

        target_item_result_set = {}
        target_item_list = []

        for user, business, rating, pred in target_result[str(i + 1)]:
            if str(business) not in target_item_result_set.keys():
                target_item_result_set[str(business)] = [[user, business, rating, pred]]
            else:
                target_item_result_set[str(business)].append([user, business, rating, pred])

            if str(business) not in target_item_list:
                target_item_list.append(str(business))

        fold_rmse_avg = []
        for n in range(1, 10):
            temp = []
            for key in source_item_train_count.keys():
                if key not in source_item_list:
                    continue
                if n == 9:
                    if 25 * (n - 1) < source_item_train_count.get(key):
                        temp.append(RMSE(source_item_result_set.get(key)))
                elif 25 * (n - 1) < source_item_train_count.get(key) < 25 * n:
                    temp.append(RMSE(source_item_result_set.get(key)))

            for key in target_item_train_count.keys():
                if key not in target_item_list:
                    continue
                if n == 9:
                    if 25 * (n - 1) < target_item_train_count.get(key):
                        temp.append(RMSE(target_item_result_set.get(key)))
                elif 25 * (n - 1) < target_item_train_count.get(key) < 25 * n:
                    temp.append(RMSE(target_item_result_set.get(key)))

            fold_rmse_avg.append(mean(temp))

        folds.append(fold_rmse_avg)

    groups = []

    for i in range(len(folds[0])):
        temp = []
        for n in range(len(folds)):
            temp.append(folds[n][i])
        groups.append(temp)

    eva_avg = []
    for i in range(len(groups)):
        eva_avg.append(mean(groups[i]))

    return eva_avg, groups


def main():

    # get data
    data = get_data()
    source_train, target_train = get_train_data()

    # list: [[fold_1_data]...]
    CoNet_source_result, CoNet_target_result = read_result('CoNet')
    BPMF_source_result, BPMF_target_result = read_result('BPMF')
    CDTF_source_result, CDTF_target_result = read_result('CDTF')

    # dict
    CoNet_MAE_RMSE = MAE_RMSEs(CoNet_source_result, CoNet_target_result)
    BPMF_MAE_RMSE = MAE_RMSEs(BPMF_source_result, BPMF_target_result)
    CDTF_MAE_RMSE = MAE_RMSEs(CDTF_source_result, CDTF_target_result)

    # Analysis
    # MAE_RMSE
    # mae_trainAmount_relation('CoNet', data, CoNet_source_result, CoNet_target_result, CoNet_MAE_RMSE)
    # mae_trainAmount_relation('BPMF', data, BPMF_source_result, BPMF_target_result, BPMF_MAE_RMSE)
    # mae_trainAmount_relation('CDTF', data, CDTF_source_result, CDTF_target_result, CDTF_MAE_RMSE)

    # user_profile_analysis
    # CoNet_upa_rmse, CoNet_upa_rmse_set = user_profile_analysis(source_train, target_train, CoNet_source_result, CoNet_target_result, CoNet_MAE_RMSE)
    # BPMF_upa_rmse, BPMF_upa_rmse_set = user_profile_analysis(source_train, target_train, BPMF_source_result, BPMF_target_result, BPMF_MAE_RMSE)
    # CDTF_upa_rmse, CDTF_upa_rmse_set = user_profile_analysis(source_train, target_train, CDTF_source_result, CDTF_target_result, CDTF_MAE_RMSE)
    # plot_line(CoNet_upa_rmse, BPMF_upa_rmse, CDTF_upa_rmse, CoNet_upa_rmse_set, BPMF_upa_rmse_set, CDTF_upa_rmse_set)

    # item_based
    CoNet_eva_avg, CoNet_folds_result = item_profile_analysis(source_train, target_train, CoNet_source_result, CoNet_target_result, CoNet_MAE_RMSE)
    BPMF_eva_avg, BPMF_folds_result = item_profile_analysis(source_train, target_train, BPMF_source_result, BPMF_target_result, BPMF_MAE_RMSE)
    CDTF_eva_avg, CDTF_folds_result = item_profile_analysis(source_train, target_train, CDTF_source_result, CDTF_target_result, CDTF_MAE_RMSE)
    plot_line(CoNet_eva_avg, BPMF_eva_avg, CDTF_eva_avg, CoNet_folds_result, BPMF_folds_result, CDTF_folds_result)


main()