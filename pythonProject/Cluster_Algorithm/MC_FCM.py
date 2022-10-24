import copy
import random
from scipy.spatial import distance
from sklearn import metrics

mL = 3.1
mU = 9.1
alpha = 0.7
Epsilon = 0.00001

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
    '''Caculate distance between two elements
    Return matrix distance of it'''
    return [[d(items[i], items[j]) for j in range(len(items))] for i in range(len(items))]
    
    # dist = []
    # for i in range(len(items)):
    #     current = []
    #     for j in range(len(items)):
    #         current.append(d(items[i],items[j]))
    #     dist.append(current)
    # return dist

def init_fuzzification_coefficient(items, number_clusters):
    '''Calculate list of fuzzification coefficient correspond with each element'''

    global mL, mU, alpha
    delta = calc_matrix_distance(items)

    # Sort matrix distance by row
    for i in range(len(delta)):
        delta[i].sort()

    delta_star = []
    n = int(len(items) / number_clusters)
    # Calculate delta_star with formula
    for i in range(len(items)):
        dummy = 0
        for j in range(n):
            dummy += delta[i][j]
        delta_star.append(dummy)

    # Find min max range of delta_star
    min_delta_star = min(delta_star)
    max_delta_star = max(delta_star)

    fuzzification_coefficient = []
    # Calculate fuzzification coefficient
    for i in range(len(items)):
        dummy = ((delta_star[i] - min_delta_star)/(max_delta_star-min_delta_star)) ** alpha
        mi = mL + (mU-mL)*dummy
        fuzzification_coefficient.append(mi)
    return fuzzification_coefficient


def init_C(items, number_clusters):
    '''Initialize random clusters '''
    C = []
    for i in range(number_clusters):
        index = random.randint(0,len(items)-1)
        C.append(items[index])
    return C

def calc_distance_item_to_cluster(items, V):
    ''' Calculate distance matrix distance between item and cluster '''
    distance_matrix = []
    for i in range(len(items)):
        current = []
        for j in range(len(V)):
            current.append(d(items[i], V[j]))
        distance_matrix.append(current)

    return distance_matrix

def update_U(distance_matrix, fuzzification_coefficient):
    '''Update membership value for each iteration'''

    U = []
    for i in range(len(distance_matrix)):
        current = []
        for j in range(len(distance_matrix[0])):
            dummy = 0
            for l in range(len(distance_matrix[0])):
                if distance_matrix[i][l] == 0:
                    current.append(0)
                    break
                dummy += (distance_matrix[i][j] / distance_matrix[i][l]) ** (2 / (fuzzification_coefficient[i] - 1))
            else:
                current.append(1/dummy)
        U.append(current)
    return U

def update_V(items, U, fuzzification_coefficient):
    ''' Update V after changing U '''

    V = []

    for k in range(len(U[0])):
        current_cluster = []

        for j in range(len(items[0])):
            dummy_sum_ux = 0.0
            dummy_sum_u = 0.0
            for i in range(len(items)):
                dummy_sum_ux += (U[i][k]**fuzzification_coefficient[i])*items[i][j]
                dummy_sum_u += (U[i][k]**fuzzification_coefficient[i])
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

def MC_FCM(items, number_clusters,max_iter = 300):
    '''Implement MC_FCM'''

    V = init_C(items,number_clusters)
    fuzzification_coefficient = init_fuzzification_coefficient(items,number_clusters)
    U = []
    for l in range(max_iter):
        distance_matrix = calc_distance_item_to_cluster(items,V)
        U = update_U(distance_matrix,fuzzification_coefficient)
        V_new = update_V(items,U,fuzzification_coefficient)
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
items= ReadData('dataset\wdbc_data.txt')
true_label = ReadLabel('dataset\wdbc_label.txt')
U,V = MC_FCM(items,2)
label = assign_label(U)
print(U,true_label,label,sep='\n')
print(metrics.rand_score(true_label,label))
