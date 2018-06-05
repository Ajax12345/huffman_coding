from collections import deque
import itertools
from typing import Dict, Any


class NotForUseError(Exception):
    pass

def create_bounds(f):
    def wrapper(cls, bounds = 26):
        return f(cls, [bounds] if isinstance(bounds, int) else bounds)
    return wrapper


def verify_char_exists(f):
    def wrapper(cls, target, path = []):
        if target not in cls:
            raise ValueError("'{}' does not exist in '{}'".format(target, cls.__class__.__name__))
        return f(cls, target, path)
    return wrapper

def check_lookup_type(**kwargs):
    def outer(f):
        def wrapper(cls, _chr):
            if type(_chr) not in kwargs.get('valid', [int, str]):
                raise TypeError("'{}' does not store values of type '{}'".format(cls.__class__.__name__, type(_chr).__name__))
            return f(cls, _chr)
        return wrapper
    return outer

class Padding:
    class max_padding:
        result = '10001'

def padd_results(ignore = False, padding = Padding.max_padding):
    def outer(f):
        def wrapper(cls, *args):
            _hashed = f(cls, *args)
            if ignore:
                return _hashed
            if getattr(padding, 'result', '1') == '10001':
                max_result = cls.deepest_route()
                return ''.join(['0']*(len(max_result)-len(_hashed)))+f(cls, *args)
            return padding+_hashed
        return wrapper
    return outer

def verify_insertion_type(f):
    def wrapper(cls, _value):
        if type(_value) not in f.__annotations__['_val']:
            raise TypeError("Expecting value of types '{}', but got '{}'".format(''.join(map(lambda x:x.__name__, f.__annotations__['_val'])), type(_value).__name__))
        f(cls, _value)
    return wrapper

def not_for_use_yet(f):
    def wrapper(*args):
        raise NotForUseError("Not for the current implementation")
    return wrapper

class NumericTree:
    def __init__(self, parent = None):
        self.left = None
        self.value = parent
        self.right = None
    @classmethod
    def flatten(cls, node):
        return [cls.flatten(node.left) if isinstance(node, cls) else getattr(node, 'left', None), getattr(node, 'value', node), cls.flatten(node.right) if isinstance(node, cls) else getattr(node, 'right', None)]
    @classmethod
    def load_Tree(cls):
        _t = NumericTree()
        for i in range(15):
            _t.insert_Val(i)
        return _t
    def _plus(self, current = 0):
        return current if not self.right or not isinstance(self.right, NumericTree) else self.right._plus(current+1)

    def _minus(self, current = 0):
        return current if not self.left or not isinstance(self.left, NumericTree) else self.left._minus(current+1)

    @staticmethod
    def tree_Depth(_all_paths):
        print(_all_paths)
        return len(max(_all_paths, key=len))

    def __eq__(self, _new_node):
        return self.value == getattr(_new_node, 'value', _new_node)
    def __lt__(self, _new_node):
        return self.value < getattr(_new_node, 'value', _new_node)
    def __gt__(self, _new_node):
        return self.value > getattr(_new_node, 'value', _new_node)
    def rotate(self, trail = []):

        if abs(getattr(self, '_minus', lambda :0)()-getattr(self, '_plus', lambda :0)()) == 2:

            if self._plus() > self._minus():
                _temp_right = self.right
                self.left = NumericTree(self.value)
                self.value = _temp_right.value
                self.right = _temp_right.right
                if trail:
                    trail[-1].right = self
            else:
                _temp_left = self.left
                self.value = _temp_left.value
                self.left = _temp_left.left.left
                self.right = NumericTree(self.value)
                if trail:
                    trail[-1].left = self
        else:
            if self.left:
                self.left.rotate(trail+[self])
            if self.right:
                self.right.rotate(trail+[self])
    def insert_Val(self, _val:[int]):
        if not self.value:
            self.value = _val
        else:
            if _val < self.value:
                if not self.left:
                    self.left = NumericTree(_val)
                else:
                    self.left.insert_Val(_val)
            else:
                if not self.right:
                    self.right = NumericTree(_val)
                else:
                    self.right.insert_Val(_val)
        self.rotate()
    def __repr__(self):
        return '<{}: left:{left}, value:{value}, right:{right}'.format(self.__class__.__name__, **self.__dict__)


class BinaryHeap(NumericTree):
    def __init__(self, temp_val = None, parent = None):
        self.left = None
        self.value = temp_val
        self.right = None
        self.parent = parent
    def __gt__(self, _node):
        return self.value > getattr(_node, 'value', _node)
    def __gt__(self, _node):
        return self.value < getattr(_node, 'value', _node)
    def __bool__(self):
        return True
    @check_lookup_type(valid=[int])
    def __getitem__(self, _char):
        return self.get_hash(_char)


    def get_hash(self, _target_char, path=[]):
        if self.left == _target_char:
            return ''.join(map(str, path+[0]))
        if self.right == _target_char:
            return ''.join(map(str, path+[1]))
        if self.right and _target_char in self.right:
            return self.right.get_hash(_target_char, path+[1])
        if self.left and _target_char in self.left:
            return self.left.get_hash(_target_char, path+[0])

    def __contains__(self, _char):
        if _char == self.value or _char == self.right or _char == self.left:
            return True
        if any(isinstance(getattr(self, i), SlotTree) for i in ['left', 'right']):
            return any(_char in getattr(self, i) for i in ['left', 'right'] if isinstance(getattr(self, i), SlotTree))

    def reassign_parent(self):
        if self.value > self.parent if self.parent else 0:
            temp_val = self.parent.value
            self.parent.value = self.value
            self.value = temp_val
            self.parent.reassign_parent()

    def insert_value(self, _value):
        if not self.value:
            self.value = _value
        else:
            if _value < self.value:
                if not self.left:
                    self.left = BinaryHeap(temp_val = _value, parent = self)
                    self.reassign_parent()
                else:
                    self.left.insert_value(_value)
            else:
                if not self.right:
                    self.right = BinaryHeap(temp_val = _value, parent = self)
                    self.reassign_parent()
                else:
                    self.right.insert_value(_value)
    @classmethod
    def scramble_range(cls, args, jump = False):
        if len(args) < 3:
            return list(args)
        a, b, c, *d = args
        return [*d, a, c, b] if not jump else [c, a, *d, b]
    @classmethod
    def partition_data(cls, args, run = 6):
        return list(itertools.chain(*[cls.scramble_range(args[i:i+run], jump = a%2 == 0) for a, i in enumerate(range(0, len(args), run))]))
    @classmethod
    def prepare_heap(cls, *args):
        _r = range(*args) if len(args) == 2 else args
        _tree = BinaryHeap()
        for i in _r:
            _tree.insert_value(i)
        return _tree
    @classmethod
    def flatten_structure(cls, d):
        return list(filter(None, [i for b in [[c] if not isinstance(c, list) else cls.flatten_structure(c) for c in d] for i in b]))



class FullNodes(object):
    __slots__ = ('_full_nodes',)
    def __init__(self):
        self._full_nodes = []

class SlotTree(NumericTree, FullNodes):
    def __init__(self, value = None, charge = 0):
        FullNodes.__init__(self)
        self.left = None
        self.value = value
        self.charge = charge
        self.right = None
        self.parent = None
    @classmethod
    @create_bounds
    def load_tree(cls, bounds=26):
        t = cls()
        for i in range(*bounds):
            t.insert_val(i)
        return t
    def __bool__(self):
        return True
    def update_root_charge(self):
        if self.parent is None:
            self.charge = not self.charge
        else:
            self.parent.update_root_charge()

    @verify_char_exists
    def _get_hash(self, _target_char, path=[]):
        if self.left == _target_char:
            return ''.join(map(str, path+[0]))
        if self.right == _target_char:
            return ''.join(map(str, path+[1]))
        if self.right and _target_char in self.right:
            return self.right.get_hash(_target_char, path+[1])
        if self.left and _target_char in self.left:
            return self.left.get_hash(_target_char, path+[0])

    def _to_dict(self) -> Dict[Any, str]:
        return {i:self._get_hash(i) for i in self._full_nodes}

    def deepest_route(self) -> str:
        return max(self._to_dict().items(), key=lambda x:len(x[-1] if x[-1] is not None else '0'))[-1]

    @verify_char_exists
    def get_hash(self, _target_char, path=[]):
        if self.value == _target_char:
            return self.deepest_route()+'1'
        if self.left == _target_char:
            return ''.join(map(str, path+[0]))
        if self.right == _target_char:
            return ''.join(map(str, path+[1]))
        if self.right and _target_char in self.right:
            return self.right.get_hash(_target_char, path+[1])
        if self.left and _target_char in self.left:
            return self.left.get_hash(_target_char, path+[0])

    def __contains__(self, _char):
        if _char == self.value or _char == self.right or _char == self.left:
            return True
        if any(isinstance(getattr(self, i), SlotTree) for i in ['left', 'right']):
            return any(_char in getattr(self, i) for i in ['left', 'right'] if isinstance(getattr(self, i), SlotTree))
    def _right_insert(self, _val, set_parent = False, node = None):
        if self.right is None:
            self.right = SlotTree(value = _val, charge = not self.charge)
            if set_parent:
                self.parent = node
            self.update_root_charge()
            self.charge = not self.charge
        else:
            self.right.insert_val(_val, set_parent = True, node = self)
    def _left_insert(self, _val, set_parent = False, node=None):
        if self.left is None:
            self.left = SlotTree(value = _val, charge = not self.charge)
            if set_parent:
                self.parent = node
            self.charge = not self.charge
            self.update_root_charge()
        else:
            self.left.insert_val(_val, set_parent = True, node=self)
    def insert_val(self, _val, set_parent = False, node=None):
        self._full_nodes.append(_val)
        if self.value is None:
            self.value = _val
        else:
            getattr(self, ['_left_insert', '_right_insert'][self.charge])(_val, set_parent = set_parent, node = node)

    @check_lookup_type(valid=[int])
    @padd_results(padding = '1')
    def __getitem__(self, _char):
        return self.get_hash(_char)


if __name__ == "__main__":

    #final_tree = BinaryHeap.prepare_heap(*BinaryHeap.partition_data(range(0, 25)))
    t = SlotTree.load_tree(6)
    print({i:t[i] for i in range(6)})
    #assert len([t[i] for i in range(26)]) == len(set([t[i] for i in range(26)]))
#print(NumericTree.flatten(t))
