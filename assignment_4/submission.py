import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    decision_tree_root = DecisionNode(None, None, lambda feature : feature[0] == 0)
    decision_tree_root.left = DecisionNode(None, None,  lambda feature : feature[3] == 0)
    decision_tree_root.right = DecisionNode(None, None, None, 1)
    
    decision_tree_root.left.left =DecisionNode(None, None, lambda feature : feature[2] == 0)
    decision_tree_root.left.right =DecisionNode(None, None, lambda feature : feature[2] == 0)
    
    decision_tree_root.left.left.left =DecisionNode(None, None, None,  1)
    decision_tree_root.left.left.right =DecisionNode(None, None, None,  0)
    
    decision_tree_root.left.right.right =DecisionNode(None, None, None,  1)
    decision_tree_root.left.right.left =DecisionNode(None, None, None, 0)
    
    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    true_pos = len(classifier_output[(classifier_output==true_labels) &(true_labels==1)])
    true_neg = len(classifier_output[(classifier_output==true_labels) & (true_labels==0)])
    false_neg = len(classifier_output[(classifier_output!=true_labels) & (true_labels==1)])
    false_pos = len(classifier_output[(classifier_output!=true_labels) & (true_labels==0)])
    
    return [[true_pos,false_neg],[false_pos,true_neg]]
    
    


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    true_pos = len(classifier_output[(classifier_output==true_labels) &(true_labels==1)])
    false_pos = len(classifier_output[(classifier_output!=true_labels) & (true_labels==0)])
    
    return true_pos/(true_pos+false_pos)
    
    
    


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    true_pos = len(classifier_output[(classifier_output==true_labels) &(true_labels==1)])
    false_neg = len(classifier_output[(classifier_output!=true_labels) & (true_labels==1)])
    
    return true_pos/(true_pos+false_neg)

    


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    true_pos = len(classifier_output[(classifier_output==true_labels) &(true_labels==1)])
    true_neg = len(classifier_output[(classifier_output==true_labels) & (true_labels==0)])
    
    
    return (true_pos+ true_neg)/len(true_labels)
    


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    class_vector = np.array(class_vector)
    num_clases = np.unique(class_vector)
    gini = 0
    for class_ in num_clases:
        p_class_ = len(class_vector[class_vector==class_])/len(class_vector)
        gini += p_class_**2
        
    
    return 1- gini


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini_index_pre = gini_impurity(previous_classes)
    
    gini_child = 0
    for classes in current_classes:
        weight = len(classes)/len(previous_classes)
        gini = gini_impurity(classes)
        gini_child += weight*gini
    
    
    return gini_index_pre -gini_child

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        max_gini = float('-inf')
        left_split = None
        left_classes = None
        right_split = None
        right_classes = None
        best_threshold = None
        best_feature = None
        num_features = len(features[0])
        #classes = np.expand_dims(classes,axis=1) #classes comes as (m,) -- > (m,1)
        #dataset = np.concatenate((features,classes),axis=1)
        if depth <= self.depth_limit:
            for i in range(num_features):
                #find the throshold: unique? equally spaced? mean? madian?
                #unique is very slow btw
                thresholds = np.unique(features[:,i])
                for threshold in thresholds:
                    #split the feaures left and right
                    left = features[features[:,i]<= threshold]
                    l_classes = classes[features[:,i]<= threshold]
                    right = features[features[:,i]< threshold]
                    r_classes = classes[features[:,i]< threshold]
                    split_features = []
                    split_features.append(l_classes)
                    split_features.append(r_classes)
                    current_gini = gini_gain(classes,split_features)
                    if current_gini > max_gini:
                        max_gini = current_gini
                        left_split= left
                        left_classes =l_classes 
                        right_split = right
                        right_classes = r_classes
                        best_threshold = threshold 
                        best_feature = i
            if max_gini> 0 : #otherwise we are trying to split a pure node (no inofromation gain)
                #split the nodes 
                root_left = self.__build_tree__(left_split,left_classes,depth+1)
                #right node
                root_right = self.__build_tree__(right_split,right_classes,depth+1)
                
                return DecisionNode(root_left,root_right,lambda feature : feature[i]<= best_threshold)
        
        #return a leaf node (the class will be the one with height number)
        class_type ,counts = np.unique(classes,return_counts=True)
        leaf_value = class_type[counts == counts.max()]
        
        return DecisionNode(None,None,None,leaf_value)
        
    
    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for i in range(len(features)):
            predicted_class = self.root.decide(features[i])
            class_labels.append(predicted_class)
            
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    raise NotImplemented()


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        # TODO: finish this.
        raise NotImplemented()


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        #return np.multiply(data,data)+data
        return data*data+data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        sum_rows = np.sum(data[:100],axis=1)
        max_row = np.max(sum_rows)
        # TODO: finish this.
        return max_row , np.where(sum_rows == max_row)[0][0]

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        flattened = np.hstack(data)
        flattened = flattened[flattened>0]
        unique, counts = np.unique(flattened, return_counts=True)
        frequencies = np.stack((unique,counts),axis=1)
        unique_dict2 = dict(frequencies)
        
        return unique_dict2.items()
        
        
def return_your_name():
    # return your name
    return "Ali Alrasheed"