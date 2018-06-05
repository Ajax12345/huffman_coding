import string, random

def traverse(on=False):
    def outer(f):
        def wrapper(cls, v):
            if not on:
                return f(cls, v)
            _v = str(v) if isinstance(v, int) else v
            if not _v[1:]:
                return '{}{}'.format(cls.row[int(_v[0])][0], BlockTree.condense_rotations(str(cls.rotation)) if cls.rotation else '')
            _block = cls.row[int(_v[0])][-1]
            if _block:
                return _block[_v[1:]]
            return BlockTree(rotation = cls.rotation+1)[_v[1:]]

        return wrapper
    return outer

class BlockTree:
    def __init__(self, _start = 0, **kwargs):
        self.rotation = kwargs.get('rotation', 0)
        self.row = [[string.ascii_lowercase[i], BlockTree(i+1, depth = kwargs.get('depth', 0)+1, rotation=self.rotation) if i+1 < 26 and kwargs.get('depth', 0) < 5 else None] for i in range(_start, (_start+6)%26)]

    @traverse(on=True)
    def __getitem__(self, _val):
        return self.row[_val] if isinstance(_val, int) else dict(self.row)[_val]
    def __len__(self):
        return 1 if not any(c for _, c in self.row) else 1+max(map(len, [c for _, c in self.row]))

    def __bool__(self):
        return True

    @classmethod
    def visualize_layer(cls, block):
        return [[a, cls.visualize_layer(b) if b is not None else b] for a, b in block]

    @staticmethod
    def condense_rotations(_r:str):
        if len(_r) == 1:
            return _r
        _results = []
        while _r:
            if len(_r) == 1:
                return ''.join(_results+list(_r))
            a, b, *c = _r
            if int(a+b) < 26:
                _results.append(string.ascii_lowercase[int(a+b)])
                _r = c
            else:
                _results.append(a)
                _r = [b]+c
        return ''.join(_results)

    def __iter__(self):
        for i in self.row:
            yield i

    def __repr__(self):
        return f'<{self.__class__.__name__}: {"|".join([a for a, _ in self.row])}'


_t = BlockTree()
print(_t[''.join(random.choice(['1', '0']) for _ in range(120))])
print(len(_t))
full_tree = BlockTree.visualize_layer(_t)
