import compression_huffman
import itertools

def burrows_wheeler(f):
    def wrapper():
        data = ''.join(b for _, b in f())
        bwt = ''.join(map(lambda x:x[-1], sorted(data[i:]+data[:i] for i in range(len(data)))))
        return ''.join([a+'F'+str(len(list(b))) for a, b in itertools.groupby(bwt)])
    return wrapper

def verify_data_types(types=[list]):
    def outer(f):
        def wrapper(cls, _data):
            if type(_data) not in types:
                raise Exception("Invalid type")
            return f(cls, _data)
        return wrapper
    return outer

def check_tree_type(f):
    def wrapper(nodes):
        if not isinstance(nodes, SortTree):
            raise TypeError("Object to be flattened must be of type '{}'".format(SortTree.__name__))
        return f(nodes)
    return wrapper

class SortTree:
    def __init__(self, _v=None):
        self.left = None
        self.parent = _v
        self.right = None
    def __eq__(self, _node):
        return self.parent == getattr(_node, 'parent', _node)
    def __lt__(self, _node):
        return self.parent < getattr(_node, 'parent', _node)
    def __ge__(self, _node):
        return self.parent >= getattr(_node, 'parent', _node)
    def __le__(self, _node):
        return self.parent <= getattr(_node, 'parent', _node)
    def insert_Val(self, _val, path = []):
        if not self.parent:
            self.parent = _val
            return [1]
        if _val.parent >= self.parent:
            if not self.right:
                self.right = _val
                return path+[1]
            else:
                return self.right.insert_Val(_val, path+[1])
        else:
            if not self.left:
                self.left = _val
                return path+[0]
            else:
                return self.left.insert_Val(_val, path+[0])
    @classmethod
    @verify_data_types(types = [str, list])
    def add_elements(cls, data):
        _tree = cls()
        for i in data:
            yield [i, ''.join(map(str, _tree.insert_Val(SortTree(i))))]
        yield _tree

    @staticmethod
    @check_tree_type
    def flatten_tree(_nodes):
        def _flattener(_node):
            return [_flattener(_node.left) if _node.left and isinstance(_node.left, SortTree) else _node.left, _node.parent if not isinstance(_node.parent, SortTree) else _flattener(_node.parent), _flattener(_node.right) if _node.right and isinstance(_node.right, SortTree) else _node.right]
        return _flattener(_nodes)

    @classmethod
    def _sorted_Result(cls, d):
        return list(filter(None, [c for b in [[i] if not isinstance(i, list) else cls._sorted_Result(i) for i in d] for c in b]))

if __name__ == '__main__':

    *result, tree = list(SortTree.add_elements('banana'))
    print(result)
    print(SortTree._sorted_Result(SortTree.flatten_tree(tree)))
    '''
    @burrows_wheeler
    def test_binary():
        return [[a, compression_huffman.Compress._to_Binary(i)] for i, a in enumerate('banana')]

    @burrows_wheeler
    def test_SortTree():
        return result

    _b = test_binary()
    _st = test_SortTree()

    print('binary: {}, length of {}'.format(_b, len(_b)))
    print('-'*15)
    print('SortTree: {}, length of {}'.format(_st, len(_st)))
    '''
