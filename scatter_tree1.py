import string, random
import math, typing
import matplotlib.pyplot as plt
import contextlib
import time
#TODO: abstract base classes for documentation purposes later on
#TODO: text graphing module
#Burrows-Wheeler transform

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

def testing_times(label = None):
    def outer(f):
        def wrapper(*args, **kwargs):
            _c = time.time()
            _result = f(*args, **kwargs)
            print(f"{label}: action completed in method '{f.__name__}' in {time.time()-_c}")
            return _result
        return wrapper
    return outer

class HashAlphabet:
    def __init__(self, max_depth = 6):
        self.max_depth = max_depth

    @testing_times(label='Random Generation')
    def __enter__(self) -> typing.List[str]:
        current = []
        for i in range(self.max_depth):
            while True:
                _row = self.__class__.random_row()
                if _row not in current:
                    current.append(_row)
                    break
        return current
    @staticmethod
    def random_row() -> str:
        _row = []
        for i in range(26):
            while True:
                _char = random.choice(string.ascii_lowercase)
                if _char not in _row:
                    _row.append(_char)
                    break
        return ''.join(_row)
    def __exit__(self, *args):
        pass

    @classmethod
    @testing_times(label='sigmoid curve')
    @contextlib.contextmanager
    def create_hashes(cls, in_full = False) -> typing.Generator[typing.Dict[int, str], None, None]:
        _r = [cls.scramble_alphabet(i) for i in range(1, 7)]
        if not in_full:
            yield _r
        else:
            yield [''.join(c[i] for i in range(26)) for c in _r]


    @classmethod
    def activate(cls, x, bounds=25, b=7, c=3, shift=0, reflect=False) -> int:
        if reflect:
            return -1*bounds/float(1+(b*pow(math.e, -1*c*(x-shift)))) + bounds
        return bounds/float(1+(b*pow(math.e, -1*c*(x-shift))))

    @classmethod
    def scramble_alphabet(cls, depth) -> typing.Dict[int, str]:
        b, c, shift = [0.1, 0.20156657963446475, 24]
        _result = {i:string.ascii_lowercase[int(cls.activate(i, b = b*depth, c = c, shift=shift))] for i in range(26)}
        missing = iter(i for i in string.ascii_lowercase if i not in _result.values())
        new_result = list(_result.items())
        return {a:b if not any(h == b and j != a for j, h in new_result[:i]) else next(missing, None) for i, (a, b) in enumerate(new_result)}

    @classmethod
    def increment_weights(cls, show_plot = False):
        def weights(d, current = []):
            if len(current) == 2:
                yield current
            else:
                for i in d:
                    yield from weights(d, current+[i])

        _start = cls.scramble_alphabet()
        b_s = range(25*50)
        c_s = [random.randint(1*b, 100*(b+1))/float(random.randint(200*(i), 300*(2*i))) for b in range(1, 25) for i in range(1, 50)]
        weighted_tests = iter([i/float(10), b, c] for i in b_s for b in c_s for c in range(1, 25))
        last = []
        while abs(len(set(_start.values())) - len(list(_start.values()))) > 4:
            _w = next(weighted_tests, None)
            #print(_w)
            if not _w:
                return _start
            b, c, shift = _w
            last = [b, c, shift]
            _start = cls.scramble_alphabet(b = b, c = c, shift=shift)
        if show_plot:
            b, c, shift = last
            plt.plot(range(500), [cls.activate(i, b = b, c = c, shift = shift) for i in range(500)])
            plt.show()
        return _start, last



with HashAlphabet.create_hashes(True) as results1:
    pass

with HashAlphabet(6) as results2:
    pass

class BlockTree:
    combo_hashes = ['11111', '11110', '11101', '11100', '11011', '11010', '11001', '11000', '10111', '10110', '10101', '10100', '10011', '10010', '10001', '10000', '01111', '01110', '01101', '01100', '01011', '01010', '01001', '01000', '00111', '00110']
    def __init__(self, _start = 0, **kwargs):
        self.rotation = kwargs.get('rotation', 0)
        self.row = [[results2[kwargs.get('depth', 0)][i], BlockTree(i+1, depth = kwargs.get('depth', 0)+1, rotation=self.rotation) if i+1 < 26 and kwargs.get('depth', 0) < 5 else None] for i in range(_start, (_start+6)%26)]

    @traverse(on=False)
    def __getitem__(self, _val):
        return self.row[_val] if isinstance(_val, int) else dict(self.row)[_val]
    def __len__(self):
        return 1 if not any(c for _, c in self.row) else 1+max(map(len, [c for _, c in self.row]))

    def __bool__(self):
        return True

    @staticmethod
    def reverse_rotations(trailing):
        return int(''.join(i if i.isdigit() else str(string.ascii_lowercase.index(i)) for i in trailing))

    @staticmethod
    def combine_binary(_input):
        #print(_input)
        _start = [_input[0]]
        _full = []
        for i in _input[1:]:
            if i != _start[-1] or (len(_start)+1 > 5 and _start[-1] == '1'):
                if len(_start)+1 < 6 or _start[-1] != i:
                    _full.append(_start)
                    _start = [i]
                else:
                    _full.append(_start)
                    _full.append(['0'])
                    _start = [i]
            else:
                _start.append(i)
        _full.append(_start)
        return ''.join(str(sum(map(int, i))) if i[0] == '1' else ''.join(i) for i in _full)


    def valid_lookups(self, letter, current = []):
        _start, *trailing = letter
        if current:
            #print(current)
            _to_check, *_trailing = self[BlockTree.combine_binary(''.join(current))]
            #print(_to_check)
        if not trailing:
            if current and _to_check == _start:
                yield current
            else:
                for option in BlockTree.combo_hashes:
                    yield from self.valid_lookups(letter, current+[option])

    def lookup_path(self, letter, rotations=0):
        pass

    def __call__(self, hashed:str):
        _target, rotations = hashed[0], self.__class__.reverse_rotations(hashed[1:])

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



#print(BlockTree.combine_binary('1111001001111111111'))
def combos(d, current = []):
    if len(current) == 5:
        yield ''.join(current)
    else:
        for i in d:
            yield from combos(d, current+[i])

#print({i:[i, _t[BlockTree.combine_binary(i)]] for i in combos(['1', '0'])})
#print(list(_t.valid_lookups('f')))
#print(HashAlphabet.scramble_alphabet(2))
'''
with HashAlphabet.create_hashes(True) as results:
    for i in results:
        print('-'*20)
        print(i)
print('\n\n')
with HashAlphabet(6) as results:
    for i in results:
        print(i)
        print('*'*20)
'''
_t = BlockTree()
print([_t[i] for i in range(6)])
