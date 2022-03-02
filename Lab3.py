import re
import math as m
import random
import pickle
import sys

"""
CSCI 630- Foundatio of Artificial Intelligence 
Author: Rutvik Pansare rp2832@g.rit.edu
"""

# method to create the feature list
def readData(fileName, type=None, algorithm="None", feature_data=None):
    f = open(fileName, "r")
    newlist = [line.rstrip() for line in f.readlines()]
    count = 0
    language = None
    for line in newlist:
        record = []
        if type.lower() == "train":
            current_line = line.split("|")
            language = current_line[0]
            sentence = current_line[1].strip().split()
        else:
            sentence = line.strip().split()
        record.append(average_word_length_greater_than_five(sentence))
        record.append(contain_Q(sentence))
        record.append(number_of_ie_greater_than_2(sentence))
        record.append(number_of_de_greater_than_1(sentence))
        record.append(number_of_Dutch_pronouns_greater_than_1(sentence))
        record.append(contains_words_unique_to_English(sentence))
        record.append(contains_words_unique_to_Dutch(sentence))
        if language:
            record.append(language)
        if algorithm.lower() == "adaboost":
            record.append(1 / len(newlist))
        feature_data.append(record)
        count = count + 1
    return feature_data


def average_word_length_greater_than_five(sentence):
    length = len(sentence)
    count = 0
    for i in sentence:
        count = count + len(i)
    average_length = count / length
    return average_length > 5


def contain_Q(sentence):
    for i in sentence:
        if "q" in i.lower():
            return True
    return False


def number_of_ie_greater_than_2(sentence):
    count = 0
    for i in sentence:
        if "ie" in i.lower():
            count = count + 1
    return count >= 2


def number_of_de_greater_than_1(sentence):
    count = 0
    for i in sentence:
        parameter = re.search("^de$", i.lower())
        if parameter is not None:
            count = count + 1
    return count >= 1


def number_of_Dutch_pronouns_greater_than_1(sentence):
    count = 0
    for i in sentence:
        parameter = re.search("(^de$)|(^hij$)|(^zij$)|(^ik$)|(^het$)|(^ze$)", i.lower())
        if parameter is not None:
            count = count + 1
    return count > 1


def contains_words_unique_to_English(sentence):
    count = 0
    for i in sentence:
        parameter = re.search("(^and$)|(^to$)|(^as$)|(^for$)|(^the$)|(^were$)|(^which$)|(^have$)|(^they$)", i.lower())
        if parameter is not None:
            count = count + 1
    return count >= 2


def contains_words_unique_to_Dutch(sentence):
    count = 0
    for i in sentence:
        parameter = re.search(
            "(^en$)|(^aan$)|(^als$)|(^voor$)|(^de$)|(^waren$)|(^die$)|(^habben$)|(^ze$)|(^het$)|(^van$)", i.lower())
        if parameter is not None:
            count = count + 1
    return count >= 1

# Node class for decision tree
class Node():
    def __init__(self, feature_index=None, left=None, right=None, threshold=None, info_gain=None, value=None,
                 amount_of_say=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.threshold = threshold
        self.info_gain = info_gain
        self.amount_of_say = amount_of_say
        # for leaf node
        self.value = value

# class for the decision tree classifier
class decision_tree_classifier():
    def __init__(self, min_sample_split=5, max_depth=8):
        self.root = Node
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def build_tree(self, current_depth=0, dataset=None):

        num_of_samples = len(dataset)
        if num_of_samples > self.min_sample_split and current_depth < self.max_depth:
            left_dataset, right_dataset, feature_index, max_gain = self.getBestSplit(dataset)
            if max_gain > 0:
                left_subtree = self.build_tree(current_depth + 1, left_dataset)
                right_subtree = self.build_tree(current_depth + 1, right_dataset)
                return Node(feature_index, left=left_subtree, right=right_subtree)

        leaf_value = self.calculateLeafValue(dataset)
        return Node(value=leaf_value)

    def calculateLeafValue(self, dataset):
        class_A_count = 0
        class_B_count = 0

        for record in dataset:
            if record[-1] == "en":
                class_A_count = class_A_count + 1
            else:
                class_B_count = class_B_count + 1
        if class_A_count > class_B_count:
            return "en"
        else:
            return "nl"

    def getBestSplit(self, dataset):
        count = 0
        total_count = 0
        left_dataSet = []
        right_dataSet = []
        attribute_Dict = self.count_stats(dataset)
        for i in dataset:
            if i[len(i) - 1] == "en":
                count = count + 1
            total_count = total_count + 1
        if count == 0 or total_count == count:
            global_entropy = 0
        else:
            global_entropy = -((count / total_count) * m.log((count / total_count), 2)) - (
                    ((total_count - count) / total_count) * m.log(((total_count - count) / total_count), 2))
        feature_index, max_gain = self.find_Importance(attribute_Dict, global_entropy)

        for record in dataset:
            if record[feature_index - 1]:
                left_dataSet.append(record)
            else:
                right_dataSet.append(record)

        return left_dataSet, right_dataSet, feature_index, max_gain

    def find_Importance(self, attribute_Dict, global_entropy):
        i = 1
        max_gain = -99999999999999999
        feature = 0
        for dictionary in attribute_Dict:
            if dictionary != 0:
                record1 = dictionary["True"]
                record2 = dictionary["False"]
                if record1[1] == 0 or record1[2] == 0:
                    entropy1 = 0
                else:
                    entropy1 = - ((record1[1] / (record1[1] + record1[2])) * m.log(
                        (record1[1] / (record1[1] + record1[2])), 2)) - (
                                       (record1[2] / (record1[1] + record1[2])) * m.log(
                                   (record1[2] / (record1[1] + record1[2])), 2))
                if record2[1] == 0 or record2[2] == 0:
                    entropy2 = 0
                else:
                    entropy2 = - ((record2[1] / (record2[1] + record2[2])) * m.log(
                        (record2[1] / (record2[1] + record2[2])), 2)) - (
                                       (record2[2] / (record2[1] + record2[2])) * m.log(
                                   (record2[2] / (record2[1] + record2[2])), 2))
                weighted_entropy = (((record1[1] + record1[2]) / (
                        record1[1] + record1[2] + record2[1] + record2[2])) * entropy1) + (((record2[1] + record2[
                    2]) / (record1[1] + record1[2] + record2[1] + record2[2])) * entropy2)
                gain = global_entropy - weighted_entropy
                if gain > max_gain:
                    max_gain = gain
                    feature = i
            i += 1
        return feature, max_gain

    def count_stats(self, newList):
        out_dict = {"True": [0, 0, 0], "False": [0, 0, 0]}
        attribute_list = [0, 0, 0, 0, 0, 0, 0]
        count = 0
        for record in newList:
            if record[-1] == "en":
                count += 1
            for i in range(len(record) - 1):
                stats = [0, 0, 0, 0]
                if attribute_list[i] == 0:
                    out_dict = {"True": [0, 0, 0], "False": [0, 0, 0]}
                    if record[i] == True:
                        stats = out_dict["True"]
                        stats[0] = stats[0] + 1
                        if record[7] == "en":
                            stats[1] = stats[1] + 1
                        else:
                            stats[2] = stats[2] + 1
                        out_dict["True"] = stats
                    else:
                        stats = out_dict["False"]
                        stats[0] = stats[0] + 1
                        if record[7] == "en":
                            stats[1] = stats[1] + 1
                        else:
                            stats[2] = stats[2] + 1
                        out_dict["False"] = stats
                    attribute_list[i] = out_dict
                else:
                    out_dict = attribute_list[i]
                    if record[i] == True:
                        stats = out_dict["True"]
                        stats[0] = stats[0] + 1
                        if record[7] == "en":
                            stats[1] = stats[1] + 1
                        else:
                            stats[2] = stats[2] + 1
                        out_dict["True"] = stats
                    else:
                        stats = out_dict["False"]
                        stats[0] = stats[0] + 1
                        if record[7] == "en":
                            stats[1] = stats[1] + 1
                        else:
                            stats[2] = stats[2] + 1
                        out_dict["False"] = stats
                    attribute_list[i] = out_dict
        return attribute_list

    def fit(self, examples):
        feature_data = []
        feature_data = readData(examples, type="train", feature_data=feature_data)
        self.root = self.build_tree(0, feature_data)

    def predict(self, testing_data):
        feature_data = []
        feature_data = readData(testing_data, type="test", feature_data=feature_data)
        predictions = [self.make_predictions(x, self.root) for x in feature_data]
        return predictions

    def make_predictions(self, x, tree):
        if tree.value != None:
            return tree.value
        if x[tree.feature_index - 1]:
            return self.make_predictions(x, tree.left)
        else:
            return self.make_predictions(x, tree.right)

# class to create decison stump
class DecisionStumps:
    def __init__(self):
        self.root = Node
        self.Dataset = []
        self.results = []

    def fit(self, feature_index, feature_data):
        self.root = self.build_tree(0, feature_data, feature_index)

    def build_tree(self, current_depth=0, dataset=None, feature_index=None):

        num_of_samples = len(dataset)
        error, results = self.calculateErrorValue(dataset, feature_index)
        if error <= 0:
            self.amount_of_say = 0
        if error > 1 and error < 1.1:
            self.amount_of_say = 0
        else:
            self.amount_of_say = 0.5 * m.log((1 - error) / (error), 2)

        self.Dataset = dataset
        self.results = results
        return Node(feature_index, amount_of_say=self.amount_of_say)

    def calculateErrorValue(self, dataset, feature_index):
        class_A_error_count = 0
        class_B_error_count = 0
        results = []
        for record in dataset:
            if record[feature_index] == False:
                if record[len(record) - 2] == "en":
                    class_B_error_count = record[-1] + class_B_error_count
                    results.append(False)
                else:
                    results.append(True)
            if record[feature_index] == True:
                if record[len(record) - 2] == "nl":
                    class_A_error_count = record[-1] + class_A_error_count
                    results.append(False)
                else:
                    results.append(True)
        return class_A_error_count + class_B_error_count, results

    def getSplit(self, dataset, feature_index):
        count = 0
        total_count = 0
        left_dataSet = []
        right_dataSet = []
        for i in dataset:
            if i[len(i) - 1] == "en":
                count = count + 1
            total_count = total_count + 1
        if count == 0 or total_count == count:
            global_entropy = 0
        else:
            global_entropy = -((count / total_count) * m.log((count / total_count), 2)) - (
                    ((total_count - count) / total_count) * m.log(((total_count - count) / total_count), 2))
        for record in dataset:
            if record[feature_index - 1]:
                left_dataSet.append(record)
            else:
                right_dataSet.append(record)

        return left_dataSet, right_dataSet

    def predict(self, testing_data_file_name):

        testing_data = readData(testing_data_file_name, type=None, algorithm="None")
        predictions = [self.make_predictions(x, self.root) for x in testing_data]
        return predictions

    def make_predictions(self, x, tree):
        if tree.value != None:
            return tree.value
        if x[tree.feature_index - 1]:
            return self.make_predictions(x, tree.left)
        else:
            return self.make_predictions(x, tree.right)


def updateWeights(dataset, results, amount_of_say):
    sample_weights = []
    pointer = 0
    new_dataset = []
    current_count = 0
    for i in range(len(dataset)):
        if results[i]:
            record = dataset[i]
            sample_weights.append(record[-1] * m.exp(amount_of_say))
        else:
            record = dataset[i]
            sample_weights.append(record[-1] * m.exp(-1 * amount_of_say))
    Sum = sum(sample_weights)
    for record in dataset:
        record[-1] = sample_weights[pointer] / Sum
        pointer = pointer + 1
    for _ in dataset:
        random_number = random.uniform(0, 1)
        for record in dataset:
            current_count = current_count + record[-1]
            if current_count > random_number:
                new_dataset.append(record)
                break

    return new_dataset

# class to create Adaboost (ensemble learning) model
class AdaBoost():

    def __init__(self, n_clf=7):
        self.n_clf = n_clf

    def fit(self, examples):
        self.clfs = []
        feature_data = []
        feature_data = readData(examples, type="train", feature_data=feature_data, algorithm="Adaboost")
        for i in range(self.n_clf):
            clf = DecisionStumps()
            clf.fit(i, feature_data)
            dataset = clf.Dataset
            results = clf.results
            amount_of_say = clf.amount_of_say
            updateWeights(dataset, results, amount_of_say)
            self.clfs.append(clf)

    def predict(self, testing_data):
        feature_data = []
        testing_data = readData(testing_data, type="test", feature_data=feature_data)
        predictions = [self.make_predictions(x) for x in testing_data]
        return predictions

    def make_predictions(self, x):
        amount_of_say_for_English = 0
        amount_of_say_for_Dutch = 0
        for stump in self.clfs:
            root = stump.root
            feature_index = root.feature_index
            if x[feature_index]:
                amount_of_say_for_English = stump.amount_of_say + amount_of_say_for_English
            else:
                amount_of_say_for_Dutch = stump.amount_of_say + amount_of_say_for_Dutch
        if amount_of_say_for_English > amount_of_say_for_Dutch:
            return "en"
        else:
            return "nl"


if sys.argv[1] == "train":
    if sys.argv[4] == "dt":
        classifier = decision_tree_classifier()
        classifier.fit(sys.argv[2])
        outfile = open(sys.argv[3], 'wb')
        pickle.dump(classifier, outfile)
        outfile.close()

    elif sys.argv[4] == "ada":
        classifier = AdaBoost()
        classifier.fit(sys.argv[2])
        outfile = open(sys.argv[3], 'wb')
        pickle.dump(classifier, outfile)
        outfile.close()
if sys.argv[1] == "predict":
    picklefile = open(sys.argv[2], 'rb')
    classifier = pickle.load(picklefile)
    picklefile.close()
    Y_pred = classifier.predict(sys.argv[3])
    for i in Y_pred:
        print(i)
