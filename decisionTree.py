from math import log2
from treeNode import TreeNode, TreeLeaf
from typing import Optional, Dict, List, Union, Any


class DecisionTree(object):
    def __init__(self, maxDepth: Optional[int] = None):
        self.maxDepth = maxDepth
        self.node = None
  
    def calc_entropy(self, y):
        y_val_cnts = {}
        for y_i in y:
            if y_i not in y_val_cnts:
                y_val_cnts[y_i] = 0
            y_val_cnts[y_i] += 1 
        
        # E(y) = -p(y==1)*lg(p(y==1)) + -p(y==0)*lg(p(y==0))
        entropy = 0
        for y_val, y_val_cnt in y_val_cnts.items():
            prob_y_val = y_val_cnt / float(len(y))
            entropy -= prob_y_val * log2(prob_y_val)
        
        return entropy

    def split_func(self, feature_value, split_value, split_type):
        if split_type == 'numerical':
            return feature_value >= split_value
        else:
            return feature_value == split_value

    def calc_split_information_gain(self, X, y, feature_name, split_value, split_type='categorical', depth=0):
        def split_func(val):
            if split_type == 'numerical':
                return val >= split_value
            else:
                return val == split_value
        x = [x_i[feature_name] for x_i in X]
        y_smaller = [y_i for x_i, y_i in zip(x, y) if not self.split_func(x_i, split_value, split_type)]
        y_larger = [y_i for x_i, y_i in zip(x, y) if self.split_func(x_i, split_value, split_type)]

        prob_smaller = len(y_smaller)/float(len(y))
        prob_larger = len(y_larger)/float(len(y))

        # E[Y|split] = #(>split)/N * E[Y|>split] + #(<=split)/N * E[Y|<=split]
        y_entropy = self.calc_entropy(y)
        y_smaller_entropy = self.calc_entropy(y_smaller)
        y_larger_entropy = self.calc_entropy(y_larger)
        # E[Y|split] = #{X >= k}/N * E[Y|X >= k] + #{X < k}/N * E[Y < k]
        y_conditional_entropy = prob_smaller*y_smaller_entropy + prob_larger*y_larger_entropy
        # IG[Y, split] = E[Y] - E[Y|split]
        split_information_gain = y_entropy - y_conditional_entropy

        split_str = f"{feature_name}{'>=' if split_type == 'numerical' else '=='}{split_value}"
        print("{}(IG[Y, {}]={:.4f}) = (E[Y]={:.4f}) - (E[Y|{}]={:.4f})".format("  "*depth, split_str, split_information_gain, y_entropy, split_str, y_conditional_entropy))
        return split_information_gain

    def find_optimal_split(self, X: List[Dict[str, Union[str, int]]], y: List[Any], depth: int = 0):
        if not X:
            raise ValueError("X is empty")
        best_info_gain = None
        best_feature_name = None
        best_feature_value = None
        best_split_type = None

        feature_names = list(X[0])
        for feature_name in feature_names:
            feature_values = set([x[feature_name] for x in X])
            for feature_value in feature_values:
                split_type = 'numerical' if type(feature_value) is int else 'categorical'
                
                num_larger = len([x for x in X if self.split_func(x[feature_name], feature_value, split_type)])
                num_smaller = len(X) - num_larger
                if num_smaller < 1 or num_larger < 1:
                    continue

                info_gain = self.calc_split_information_gain(X, y, feature_name, feature_value, split_type, depth=depth)
                if not best_info_gain or info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_name = feature_name
                    best_feature_value = feature_value
                    best_split_type = split_type
                
        
                
        #print(' .     best_feature_name={}, X={}'.format(best_feature_name, X))
        if not best_feature_name:
            # no distinct feature values
            return None, None, None

        num_larger = len([x for x in X if self.split_func(x[best_feature_name], best_feature_value, best_split_type)])
        num_smaller = len(X) - num_larger
        print("{}Best split: {}(IG[Y, {}]={:.4f}), larger={}, smaller={}".format("  "*depth, best_feature_name, best_feature_value, best_info_gain, num_larger, num_smaller))
        return best_feature_name, best_feature_value, best_split_type

    def train(self, X, y, curr_depth=0):
        print("{}train(X={}, y={})".format("  "*curr_depth, X, y))
        if len(X) == 1:
            return TreeLeaf(y[0])

        split_feature_name, split_value, split_type = self.find_optimal_split(X, y, depth=curr_depth)
        if not split_feature_name:
            # no distinct feature values found
            return TreeLeaf(y[0])

        X_smaller = [x_i for x_i in X if not self.split_func(x_i[split_feature_name], split_value, split_type)]
        y_smaller = [y_i for x_i, y_i in zip(X, y) if not self.split_func(x_i[split_feature_name], split_value, split_type)]

        X_larger = [x_i for x_i in X if self.split_func(x_i[split_feature_name], split_value, split_type)]
        y_larger = [y_i for x_i, y_i in zip(X, y) if self.split_func(x_i[split_feature_name], split_value, split_type)]

        smaller_node = self.train(X_smaller, y_smaller, curr_depth+1)
        larger_node = self.train(X_larger, y_larger, curr_depth+1)
        node = TreeNode(split_feature_name, split_value, split_type, smaller_child=smaller_node, larger_child=larger_node)

        if curr_depth == 0:
            self.node = node

        return node

    def predict_single(self, x):
        node = self.node
        while True:
            if node.is_leaf:
                return node.label
            if self.split_func(x[node.split_feature_column], node.split_feature_value, node.split_type):
                node = node.larger_child
            else:
                node = node.smaller_child
            
    def predict(self, X):
        if type(X) is List:
            return [self.predict_single(x) for x in X]
        else:
            return self.predict_single(X)
