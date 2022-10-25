import copy
import random
from scipy.spatial import distance
from sklearn import metrics

m = 6
Epsilon = 0.0001

def ReadData(fileName):
    # Read the file, splitting by lines
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()

    items = []
    for i in range(len(lines)):
        itemFeatures = list(map(float, lines[i].split(",")))
        items.append(itemFeatures)

    # for i in range(len(lines)):
    #     line = lines[i].split(',')
    #     itemFeatures = []

    #     for j in range(len(line)):
    #         # Convert feature value to float
    #         v = float(line[j])
    #         # Add feature value to dict
    #         itemFeatures.append(v)

    #     items.append(itemFeatures)
    
    return items

def ReadLabel(fileName):
    # Read the file, splitting by lines
    f = open(fileName, 'r')
    lines = f.read().splitlines()
    f.close()

    # true_label = []

    # for i in range(len(lines)):
    #     true_label.append(lines[i])
    # return true_label
    return lines


def d(i,j):
    '''Calculate Euclidean'''
    return distance.euclidean(i,j)

def calc_matrix_distance(items):
    '''Calculate distance between two elements
    Return matrix distance of it'''
    return [[d(items[i], items[j]) for j in range(len(items))] for i in range(len(items))]
    # dist = []
    # for i in range(len(items)):
    #     current = []
    #     for j in range(len(items)):
    #         current.append(d(items[i],items[j]))
    #     dist.append(current)
    # return dist


def init_C(items, number_clusters):
    '''Initialize random clusters '''
    C = []
    for i in range(number_clusters):
        index = random.randint(0,len(items)-1)
        C.append(items[index])
    return C

def calc_distance_item_to_cluster(items, V):
    ''' Calculate distance matrix distance between item and cluster '''
    distance_matrix = [[d(items[i], V[j]) for j in range(len(V))] for i in range(len(items))]
    # distance_matrix = []
    # for i in range(len(items)):
    #     current = []
    #     for j in range(len(V)):
    #         current.append(d(items[i], V[j]))
    #     distance_matrix.append(current)

    return distance_matrix

def update_U(distance_matrix):
    '''Update membership value for each iteration'''
    global m
    U = []
    for vector in distance_matrix:
        current = []
        for j in range(len(vector)):
            dummy = 0
            for l in range(len(vector)):
                if vector[l] == 0:
                    current.append(0)
                    break
                dummy += (vector[j] / vector[l]) ** (2 / (m - 1))
            else:
                current.append(1/dummy)
        U.append(current)
    
    
    # for i in range(len(distance_matrix)):
    #     current = []
    #     for j in range(len(distance_matrix[0])):
    #         dummy = 0
    #         for l in range(len(distance_matrix[0])):
    #             if distance_matrix[i][l] == 0:
    #                 current.append(0)
    #                 break
    #             dummy += (distance_matrix[i][j] / distance_matrix[i][l]) ** (2 / (m - 1))
    #         else:
    #             current.append(1/dummy)
    #     U.append(current)
    return U

def update_V(items, U):
    ''' Update V after changing U '''
    global m
    V = []

    for k in range(len(U[0])):
        current_cluster = []

        for j in range(len(items[0])):
            dummy_sum_ux = 0.0
            dummy_sum_u = 0.0
            for i in range(len(items)):
                dummy_sum_ux += (U[i][k]**m)*items[i][j]
                dummy_sum_u += (U[i][k]**m)
            current_cluster.append(dummy_sum_ux/dummy_sum_u)
        V.append(current_cluster)

    return V

def end_condition(V_new,V):
    ''' End condition '''

    global Epsilon
    for i in range(len(V)):
        if d(V_new[i],V[i]) > Epsilon:
            return False
    return True

def FCM(items, number_clusters,max_iter = 3000):
    '''Implement FCM'''

    V = init_C(items,number_clusters)
    U = []

    for l in range(max_iter):
        distance_matrix = calc_distance_item_to_cluster(items,V)
        U = update_U(distance_matrix)
        V_new = update_V(items,U)
        if end_condition(V_new,V):
            break
        V = copy.deepcopy(V_new)

    return U,V

def assign_label(U):
    label = []

    for i in range(len(U)):
        maximum = max(U[i])
        max_index = U[i].index(maximum)
        label.append(max_index)

    return label


items= ReadData('pythonProject\dataset\wdbc_data.txt')
true_label = ReadLabel('pythonProject\dataset\wdbc_label.txt')

U,V = FCM(items,2)
label = assign_label(U)
print(U,true_label,label,sep='\n')
print(metrics.rand_score(true_label,label))