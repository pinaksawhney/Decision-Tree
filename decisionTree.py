import sys
import math
from collections import Counter
import csv
import numpy as np

def data_preprocessing(inFile):
    f = open(inFile, 'r')
    data = []
    p = csv.DictReader(f, delimiter='\t')
    for _ in p:
        data.append(_)
    f.close()
    return data

def giveLabel(inFile):
    f = open(inFile, 'r')
    label = list(csv.reader(f, delimiter='\t'))[0][-1]
    return label

def entropy_helper(lst):
    count_val = Counter(lst).values()
    p1 = count_val[0] / (1.0 * sum(count_val))
    p2 = count_val[1] / (1.0 * sum(count_val))
    try:
        e1 = p1 * (math.log(p1, 2))
    except ValueError:
        e1 = 0
    try:
        e2 = p2 * (math.log(p2, 2))
    except:
        e2 = 0

    entropy = -(e1 + e2)
    return entropy


def helper_Mutual_information(data, Y, attribute_name):
    lst_Y = []
    lst_X = []
    for _ in data:
        lst_Y.append(_[Y])
    H_Y = entropy_helper(lst_Y)

    for _ in data:
        lst_X.append(_[attribute_name])
    A_values = np.unique(lst_X)
    Y_values = np.unique(lst_Y)


    P_A_0 = Counter(lst_X).values()[0]/(1.0*sum(Counter(lst_X).values()))
    P_A_1 = 1-P_A_0

    # if y=0 and a=0 ++
    cnt = 0
    total = 0
    for _ in data:
        if _[Y] == Y_values[0] and _[attribute_name] == A_values[0]:
            cnt = cnt+1

    for _ in data:
        if _[attribute_name] == A_values[0]:
            total = total+1

    P_Y_0_A_0 = cnt/(1.0*total)

    # if y=1 and a=0 ++
    P_Y_1_A_0 = 1-P_Y_0_A_0


    # if y=0 and a=1 ++
    cnt = 0
    total = 0
    for _ in data:
        if _[Y] == Y_values[0] and _[attribute_name] == A_values[1]:
            cnt = cnt + 1

    for _ in data:
        if _[attribute_name] == A_values[1]:
            total = total + 1

    P_Y_0_A_1 = cnt / (1.0 * total)

    # if y=0 and a=1 ++
    P_Y_1_A_1 = 1 - P_Y_0_A_1

    try:
        a = math.log(P_Y_1_A_0, 2)
    except ValueError:
        a = 0

    try:
        b = math.log(P_Y_0_A_0, 2)
    except ValueError:
        b = 0

    try:
        c = math.log(P_Y_1_A_1, 2)
    except ValueError:
        c = 0

    try:
        d = math.log(P_Y_0_A_1, 2)
    except ValueError:
        d = 0

    H_Y_A_0 = -((P_Y_0_A_0)*(b) + (P_Y_1_A_0)*(a))
    H_Y_A_1 = -((P_Y_0_A_1)*(d) + (P_Y_1_A_1)*(c))
    H_Y_A = (P_A_0*H_Y_A_0) + (P_A_1*H_Y_A_1)
    I_Y_A = H_Y - H_Y_A
    return I_Y_A


def error_rate_helper():
    count_val = 0
    error_rate = min(count_val) / (1.0 * sum(count_val))
    return error_rate


def train_tree(data, Y):
    I_lst = []
    attribs = data[0].keys()
    Y_ind = data[0].keys().index(Y)
    del (attribs[Y_ind])
    for _ in attribs[:len(attribs)]:
        i = helper_Mutual_information(data, Y, _)
        I_lst.append(i)
    split_attribute = attribs[I_lst.index(max(I_lst))]
    lst_X = []
    lst_Y = []
    for _ in data:
        lst_X.append(_[split_attribute])
    for _ in data:
        lst_Y.append(_[Y])
    A_values = np.unique(lst_X)
    Y_values = np.unique(lst_Y)
    y_0 = 0
    y_1 = 0
    trained = {}
    for _ in data:
        if _[split_attribute] == A_values[0]:
            if _[Y] == Y_values[0]:
                y_0 += 1
            if _[Y] == Y_values[1]:
                y_1 += 1
    trained.update({A_values[0]: {Y_values[0]: y_0,  Y_values[1]: y_1}})
    y_0 = 0
    y_1 = 0
    for _ in data:
        if _[split_attribute] == A_values[1]:
            if _[Y] == Y_values[0]:
                y_0 += 1
            if _[Y] == Y_values[1]:
                y_1 += 1
    trained.update({A_values[1]: {Y_values[0]: y_0,  Y_values[1]: y_1}})
    model = {split_attribute: trained}
    return model


class Tree:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


def get_new_Node(val):
    newNode = Tree(val)
    return newNode


def add_Child(root, val):
    if root == None:
        root = get_new_Node(val)

    elif root.val >= val:
        root.left = add_Child(root.left, val)

    elif root.val < val:
        root.right = add_Child(root.right, val)

    return root


def printInorder(tree_vals, root):
     if root != None:
         printInorder(tree_vals, root.left)
         try:
             print root.val, tree_vals[tree_vals.keys()[0]][root.val]
         except KeyError:
             print '\n'
             pass
         finally:
            printInorder(tree_vals, root.right)


if __name__ == '__main__':
    # inputs from cli
    in_TrainFile = sys.argv[1]
    in_TestFile = sys.argv[2]
    max_depth = sys.argv[3]
    out_train_labels = sys.argv[4]
    out_test_labels = sys.argv[5]
    metrics = sys.argv[6]

    # preprocess data and get a list
    data = data_preprocessing(in_TrainFile)
    label = giveLabel(in_TrainFile)

    #calculate Mutual info for all attributes; select the one that has max mutual info and split on it
    model = train_tree(data, label)
    # after 1st split model dict is returned

    #print tree
    root_val = model.keys()[0]
    child_vals = model[root_val].keys()
    root = Tree(root_val)
    root = add_Child(root, child_vals[0])
    root = add_Child(root, child_vals[1])
    printInorder(model, root)

    y_or_no = model[model.keys()[0]].keys()
    classified_value_y = model[model.keys()[0]][y_or_no[0]]
    classified_value_n = model[model.keys()[0]][y_or_no[1]]

    print classified_value_y






    #for _ in classified_value_n
    #if 0 in classified_value_n.values():
    #    zero_index = index
    #    print max
        # stop tree
        # prediction done

    #print classified_values






