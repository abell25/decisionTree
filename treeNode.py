class TreeNode(object):
    def __init__(self, split_feature_column, split_feature_value, split_type, smaller_child, larger_child):
        self.split_feature_column = split_feature_column
        self.split_feature_value = split_feature_value
        self.split_type = split_type
        self.smaller_child = smaller_child
        self.larger_child = larger_child

        self.label = None
        self.is_leaf = False

    def print(self, depth=0):
        if self.is_leaf:
            print("  "*depth + "Leaf: {}".format(self.label))
        else:
            split_type_str = ">=" if self.split_type == 'numerical' else "=="
            print("  "*depth + "Split on {} {} {}".format(self.split_feature_column, split_type_str, self.split_feature_value))
            self.larger_child.print(depth+1)
            self.smaller_child.print(depth+1)

class TreeLeaf(TreeNode):
    def __init__(self, label):
        super().__init__(split_feature_column=None, split_feature_value=None, split_type=None, smaller_child=None, larger_child=None)
        self.label = label
        self.is_leaf = True


