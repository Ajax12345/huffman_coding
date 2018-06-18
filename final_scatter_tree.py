import string, random
import math, typing
import matplotlib.pyplot as plt
import contextlib
import time
from collections import defaultdict
#TODO: abstract base classes for documentation purposes later on
#TODO: if using comb_hashes with perm({'1', '0'}, 5), find nodes with lower likelyhood of being seen. Thus, a greater hash value can be assigned
#Burrows-Wheeler transform

def node_number(_):
    def wrapper(cls):
        return getattr(cls, '_full_node_count')()
    return wrapper

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
    def full_character_range_scramble(cls):
        _results = []
        for _ in range(52):
            while True:
                _r = random.choice(string.ascii_lowercase+string.ascii_uppercase)
                if _r not in _results:
                    _results.append(_r)
                    break
        return _results

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


'''
with HashAlphabet.create_hashes(True) as results1:
    _results = ''.join(results1)
    results1 = [_results[i:i+6] for i in range(0, len(_results), 6)]
    print('results1', results1)
'''
'''
with HashAlphabet(6) as results2:
    _results = ''.join(results2)
    results2 = [_results[i:i+6] for i in range(0, len(_results), 6)]
'''
from results2_scrambled_alphabet import results2

results1 = [['r', 'z', 'g', 'G', 'Q', 'o'], ['K', 'J', 'j', 'W', 'T', 'H'], ['N', 'e', 'Y', 'v', 'F', 'M'], ['w', 'f', 'n', 'R', 'P', 'V'], ['C', 'I', 'x', 'd', 'B', 'S'], ['b', 'u', 'Z', 'U', 'c', 'k'], ['p', 'E', 's', 'D', 'X', 'h'], ['q', 'm', 'a', 'l', 'i', 'A'], ['t', 'L', 'y', 'O']]

class BlockTree:
    current_depth = 0
    seen_result = False
    frequencies = defaultdict(int)
    depths = defaultdict(list)
    combo_hashes = ['1', '11', '111', '1111', '11111', '11110', '1110', '11101', '11100', '110', '1101', '11011', '1100', '101', '10110', '10101', '10100', '100', '1001', '10011', '10001', '011', '01111', '01110', '0111']
    def __init__(self, _start = 0, **kwargs):
        self.rotation = kwargs.get('rotation', 0)
        #self.row = [[results2[kwargs.get('depth', 0)][i], BlockTree(i+1, depth = kwargs.get('depth', 0)+1, rotation=self.rotation) if i+1 < 26 and kwargs.get('depth', 0) < 5 else None] for i in range(_start, (_start+6)%26)]
        #print(results2[kwargs.get('new_depth', 0)])
        BlockTree.current_depth += 1
        self.row = [[c, self.__class__._get_frequency(c), BlockTree(depth = kwargs.get('depth', 0)+1, rotation=self.rotation, new_depth = kwargs.get('new_depth', 0)+1) if kwargs.get('depth', 0) < 5 else None] for c in results1[BlockTree.current_depth%len(results1)]]
    
    def get_frequencies(self, target):
        return sum((a == target)+getattr(b, 'get_frequencies', lambda _:0)(target) for a, _, b in self.row)
    
    @traverse(on=True)
    def __getitem__(self, _val):
        return self.row[_val] if isinstance(_val, int) else {a:b for a, _, b in self.row}[_val]
    
    @classmethod
    def _get_frequency(cls, target:str):
        cls.frequencies[target] += 1
        return cls.frequencies[target]

    def set_max_depths(self, depth = 1):
        for a, _, b in self.row:
            BlockTree.depths[a].append(depth)
            getattr(b, 'set_max_depths', lambda *_:None)(depth+1)

    @classmethod
    def set_maximums(cls):
        cls.depths = {a:max(b) for a, b in cls.depths.items()}

    @node_number
    def __len__(self):
        return 1 if not any(c for _a, _, c in self.row) else 1+max(map(len, [c for _a, _, c in self.row]))
    
    def _full_node_count(self):
        return sum(1+getattr(b, '_full_node_count', lambda :0)() for _a, _, b in self.row)
    
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

    def _lookup_result(self, target, current = []):
        #print('current depth counter', self.__class__.current_depth)
        #print(self)
        
        if current and self[BlockTree.combine_binary(''.join(current))] == target:
            yield current
            BlockTree.seen_result = True
        else:
            if not BlockTree.seen_result:
                for i in self.__class__.combo_hashes:
                    print('in loop')
                    _current = self[BlockTree.combine_binary(''.join(current+[i]))]
                    if not target[1:]:
                        print('first if')
                        if not _current[1:]:
                            print('first inner if')
                            yield from self._lookup_result(target, current+[i])
                    else:
                        print('first else')
                        if not _current[1:]:
                            print('first else if')
                            yield from self._lookup_result(target, current+[i])
                        else:
                            print('first else else')
                            if int(BlockTree.reverse_rotations(_current[1:])) < int(BlockTree.reverse_rotations(target[1:])):
                                print('first else else if')
                                yield from self._lookup_result(target, current+[i])
            
            

    def lookup_result(self, target:str, current = []) -> typing.Generator[typing.List[str], None, None]:
        #IDEA: could brute-force all options in rotation range
        print(current)
        if (lambda x:'' if not x else self[self.__class__.combine_binary(''.join(x))])(current) == target:
            yield current
        else:
            for i in self.__class__.combo_hashes:
                if not target[1:]:
                    _test = self[self.__class__.combine_binary(''.join(current+[i]))]
                    if not _test[1:]:
                        yield from self.lookup_result(target, current+[i])
                else:
                    _, *trailing = self[self.__class__.combine_binary(''.join(current+[i]))]
                    if not trailing:
                        yield from self.lookup_result(target, current+[i])
                    else:
                        rotations = int(self.__class__.reverse_rotations(''.join(trailing)))
                        check_against = int(self.__class__.reverse_rotations(target[1:]))
                        if rotations < check_against:
                            yield from self.lookup_result(target, current+[i])
        

  

    def __call__(self, hashed:str) -> list:
        return list(self._lookup_result(hashed))

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
        return f'<{self.__class__.__name__}: {"|".join([a for a, *_ in self.row])}'

                
if __name__ == '__main__':

    def options(d:str, current = []):
        _options = [i for i in BlockTree.combo_hashes if d.startswith(i) and (not d[len(i):] or any(d[len(i):].startswith(c) for c in BlockTree.combo_hashes))]
        #print(_options)
        for i in _options:
            if not d[len(i):]:
                yield current+[i]
            else:
                yield from options(d[len(i):], current+[i])
    


    _t = BlockTree()
    '''
    print({i:_t.get_frequencies(i) for i in string.ascii_lowercase})
    print(len(_t))
    
    _nums = [0, 4, 1, 4, 1]
    _hashing = ''.join(BlockTree.combo_hashes[i] for i in _nums)
    print(_hashing)
    _hashed = _t[BlockTree.combine_binary(_hashing)]
    print(_hashed)
    '''
  
    print(_t[BlockTree.combine_binary('1'*12)])

 


    #print(_t.__class__.combo_hashes)
    #print(list(options('1111011')))
    #print('starting row', _t)
    #t = _t(_hashed)
    #print(t)
    
    #print({''.join(i):[_t[_t.__class__.combine_binary(''.join(i))], list(options(''.join(i)))] for i in t})
    

   
