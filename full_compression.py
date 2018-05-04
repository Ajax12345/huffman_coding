import re, typing, string
#written in Python3
import datetime, copy
from collections import Counter
import functools, os
import array, contextlib
import itertools, time

class InvalidFileType(TypeError):
    pass

class IndexResult:
    pass

class _TestingDecodeStreamError(Exception):
    def __str__(self):
        return "error with 'decode_stream'"

def time_compression(f):
    def wrapper(*args):
        c = time.time()
        f(*args)
        print('{}ion completed in {} seconds'.format(f.__name__, abs(c-time.time())))
    return wrapper

def verify_binary_type(f):
    def wrapper(cls, freq):
        if not isinstance(freq, f.__annotations__['val']):
            raise TypeError("Expecting an integer value for '{}'".format(f.__name__))
        return f(cls, freq)
    return wrapper

test_token = True

def verify_test(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not test_token:
            raise TypeError("Operation not permitted")
        return f(*args, **kwargs)
    return wrapper

def verify_bin_file(f):
    @functools.wraps(f)
    def wrapper(_, filename):
        if not re.findall('\.bin$', filename):
            raise ValueError("expecting '.bin' filename for '{}'".format(f.__name__))
        return f(_, filename)
    return wrapper

def verify_read_bin_file(f):
    def wrapper(filename):
        if not re.findall('^\d+[\w\W]+\.bin$', filename):
            raise InvalidFileType("'{}' does not look like something we compressed".format(filename))
        return f(filename)
    return wrapper

class TreeNode:
    def __init__(self, *args):
        self.left, self.right = args
        self.parent = sum(getattr(i, 'parent', i[-1] if isinstance(i, tuple) else None) for i in args)
    def __lt__(self, _parent):
        return self.parent < getattr(_parent, 'parent', None if isinstance(_parent, TreeNode) else _parent[-1])
    def __gt__(self, _parent):
        return self.parent > getattr(_parent, 'parent', None if isinstance(_parent, TreeNode) else _parent[-1])
    def __getitem__(self, *args):
        return IndexResult
    def lookup(self, target, _path = None):
        if not _path:
            _path = []
        if not isinstance(self.right, TreeNode):

            if self.right and self.right[0] == target:
                return _path+[1]
        if not isinstance(self.left, TreeNode):
            if self.left and self.left[0] == target:
                return _path+[0]
        if self.left and not isinstance(self.left, tuple) and self.has_target(self.left, target):
            return self.left.lookup(target, _path+[0])
        return self.right.lookup(target, _path+[1])
    def __bool__(self):
        return True
    def has_target(self, _t, target):
        if _t.left and _t.left[0] == target:
            return True
        if _t.right and _t.right[0] == target:
            return True
        vals = []
        if _t.right and not isinstance(_t.right, tuple):
            vals.append(_t.has_target(_t.right, target))
        if _t.left and not isinstance(_t.left, tuple):
            vals.append(_t.has_target(_t.left, target))
        return any(vals)

    @classmethod
    def flatten(cls, node) -> typing.List:
        return [cls.flatten(node.left) if isinstance(node.left, TreeNode) else node.left, node.parent, cls.flatten(node.right) if isinstance(node.right, TreeNode) else node.right]


def valid_file_extensions(**kwargs):
    def outer(f):
        @functools.wraps(f)
        def wrapper(cls, filename):
            if kwargs.get('include', True):
                if re.findall('(?<=\.)\w+$', filename)[0] not in kwargs.get('extensions', []):
                    raise InvalidFileType("Expecting file of type {}, not {}".format(', '.join('.'+i for i in kwargs.get(extensions)), re.findall('(?<=\.)\w+$')[0]))
            else:
                if re.findall('(?<=\.)\w+$', filename)[0] in kwargs.get('extensions', []):
                    raise InvalidFileType("File type cannot be of {}".format(', '.join('.'+i for i in kwargs.get('extensions'))))
            return f(cls, filename)
        return wrapper
    return outer
def valid_file_name(f):
    @functools.wraps(f)
    def wrapper(cls, filename):
        if not re.findall('^\d+[\w_\.]+$', filename):
            raise InvalidFileType("'{}' does not look like something we compressed".format(filename))
        return f(cls, filename)
    return wrapper

class Decompress:
    @valid_file_extensions(extensions = ['txt', 'csv', 'py', 'bin'])
    @valid_file_name
    def __init__(self, filename):
        self.filename = filename
        #self.file_contents = open(self.filename).read()
    @staticmethod
    def read_compressed_file(content) -> typing.List:
        keys = '\n'.join(i for i in content if any(c in i for c in string.ascii_letters+string.punctuation+string.whitespace))
        compressed = ''.join(i for i in content if re.findall('^[01]+$', i))
        results = re.findall('[a-zA-Z\W_]|\d+', keys)
        return compressed, sorted([(results[i], int(results[i+1])) for i in range(0, len(results), 2)], key=lambda x:x[-1])

    @staticmethod
    def _master_header_Parser(content):
        header_hashed = Compress._hash_Alphabet()
        _hashed = {b:a for a, b in header_hashed.items()}
        header = []
        while True:
            possibilities = [i for i in _hashed if content.startswith(i)]
            assert len(possibilities) <= 1, "Character error. Ensure that all chars are ascii"
            if not possibilities:
                break
            else:
                if _hashed[possibilities[0]] == 'F':
                    #print([i for i in _hashe if content[len(possibilities[0]):].startswith(i)])
                    if 'F' in header or not _hashed[[i for i in _hashed if content[len(possibilities[0]):].startswith(i)][0]].isdigit():
                        content = content[len(possibilities[0]):]
                        break
                header.append(_hashed[possibilities[0]])
                content = content[len(possibilities[0]):]
        return content, (lambda y:{y[i]:y[i+1] for i in range(0, len(y), 2)})([(lambda x:x[0] if not a else int(''.join(x)))(list(b)) for a, b in itertools.groupby(header, key=lambda x:x.isdigit())])
    def __enter__(self):
        source, counts = Decompress._master_header_Parser(Decompress._read_binary_File(self.filename))
        _counts = sorted(counts.items(), key=lambda x:x[-1])
        frequencies = copy.deepcopy(_counts)
        print('counts in compression here', _counts)
        print('frequencies here', frequencies)
        while len(frequencies) > 1:
            a, b, *c = frequencies
            frequencies = sorted(c+[TreeNode(a, b)])
        final_lookup = {''.join(map(str, frequencies[0].lookup(i))):i for i, _ in _counts}
        print('locations new', final_lookup)
        print('decoded info', ''.join(Decompress._decode_stream(source, final_lookup)))
        #print('decoded info', ''.join(Decompress.decode_stream(source, final_lookup)))
    def __exit__(self, *args):
        pass


    @staticmethod
    def _decode_stream(message, encoder):
        print('encoder', encoder)
        seen = []
        while message:
            possibilities = [i for i in encoder if message.startswith(i)]
            if not possibilities:
                print('message currently', message)
                yield ''
            if len(possibilities) == 1:
                message = message[len(possibilities[0]):]
                yield encoder[possibilities[0]]
            else:
                try:
                    if not any(message[len(possibilities[-1]):].startswith(i) for i in encoder):
                        message = message[len(possibilities[0]):]
                        yield encoder[possibilities[0]]
                except IndexError:
                    break
                else:
                    raise _TestingDecodeStreamError


    @staticmethod
    @verify_read_bin_file
    def _read_binary_File(filename):
        _result = ''
        with open(filename, 'rb') as f:
            byte = f.read(1)
            while byte:
                byte = ord(byte)
                _result += bin(byte)[2:].rjust(8, '0')
                byte = f.read(1)
        return _result

    @classmethod
    def decode_stream(cls, message, encoder, _current_message = None):
        '''this decode stream raises a recursion depth exceeded error'''
        if _current_message is None:
            _current_message = []
        if not message:
            return _current_message
        possibilities = [i for i in encoder if message.startswith(i)]
        if not possibilities:
            raise _TestingDecodeStreamError
        if len(possibilities) == 1:
            return Decompress.decode_stream(message[len(possibilities[0]):], encoder, _current_message+[encoder[possibilities[0]]])
        if not any(message[len(possibilities[0]):].startswith(c) for c in encoder):
            return Decompress.decode_stream(message[len(possibilities[0]):], encoder, _current_message+[encoder[possibilities[0]]])
        return Decompress.decode_stream(message[len(possibilities[-1]):], encoder, _current_message+[encoder[possibilities[-1]]])
class Compress:
    ascii_alphabet = string.ascii_letters+string.punctuation+string.whitespace+''.join(map(str, range(10)))
    @valid_file_extensions(extensions=['txt', 'csv', 'py'])
    def __init__(self, filename):
        self.filename = filename
        self._file_data = open(filename).read()
    def __enter__(self):
        frequencies = Compress.get_counts(self._file_data)
        _frequencies = copy.deepcopy(frequencies)
        while len(frequencies) > 1:
            a, b, *c = frequencies
            frequencies = sorted(c+[TreeNode(a, b)])
        print('parent value: ', frequencies[0].parent)
        print(TreeNode.flatten(frequencies[0]))
        full_hashing = dict([(i, ''.join(map(str, frequencies[0].lookup(i)))) for i in set(re.findall('[\w\W]', self._file_data))])
        #print('full hashing', full_hashing)
        _hashed_alphabet = Compress._hash_Alphabet()
        #print('hashed alphabet', _hashed_alphabet)
        temp_header = ''.join(_hashed_alphabet[a]+''.join(_hashed_alphabet[c] for c in str(b)) for a, b in _frequencies)
        _converted_content = '{}1100101{}'.format(''.join(_hashed_alphabet[a]+''.join(_hashed_alphabet[c] for c in str(b)) for a, b in _frequencies), ''.join(full_hashing[i] for i in self._file_data))

        Compress._write_bin_file(Compress._to_Bytes(_converted_content), '{}.bin'.format(''.join(str(getattr(datetime.datetime.now(), i)) for i in ['day', 'month', 'year', 'second', 'minute', 'hour'])+re.sub('\.[a-zA-Z]+$', '', self.filename)))
        '''
        with Compress._main_file_creation('{}1100101{}'.format(''.join(_hashed_alphabet[a]+''.join(_hashed_alphabet[c] for c in str(b)) for a, b in _frequencies), ''.join(full_hashing[i] for i in self._file_data)), self.filename) as f:
            pass

        with open('{}{}'.format(''.join(str(getattr(datetime.datetime.now(), i)) for i in ['day', 'month', 'year', 'minute', 'hour', 'second']), self.filename), 'a') as f:
            f.write('{}\n{}\n'.format(''.join(a+str(b) for a, b in _frequencies), ''.join(full_hashing[i] for i in self._file_data)))

        with Compress._main_file_creation('{}\n{}\n'.format(''.join(a+str(b) for a, b in _frequencies), ''.join(full_hashing[i] for i in self._file_data)), self.filename) as f:
            pass
        '''

        #print(Compress.print_structure(TreeNode.flatten(frequencies[0])))
    def __exit__(self, *args):
        pass

    @staticmethod
    def _to_Bytes(data):
        b = bytearray()
        for i in range(0, len(data), 8):
            b.append(int(data[i:i+8], 2))
        return bytes(b)

    @classmethod
    def _to_Binary(cls, value:int, current = None):
        if not value:
            return '0' if not current else ''.join(map(str, current))
        _val = max(i for i in range(value) if pow(2, i) <= value) if value else 0

        return cls._to_Binary(value-pow(2, _val), [1]+([0]*_val) if not current else [1 if len(current)-1 - i == _val else a for i, a in enumerate(current)])

    @classmethod
    def _hash_Alphabet(cls):
        frequencies = cls.get_counts(cls.ascii_alphabet)
        print('starting frequencies in hashing', frequencies)
        _frequencies = copy.deepcopy(frequencies)
        print('copied frequencies', _frequencies)
        while len(frequencies) > 1:
            a, b, *c = frequencies
            frequencies = sorted(c+[TreeNode(a, b)])
        full_hashing = dict([(i, ''.join(map(str, frequencies[0].lookup(i)))) for i in cls.ascii_alphabet])
        return full_hashing

    @staticmethod
    @verify_test
    @verify_bin_file
    def _write_bin_file(content, filename):
        with open(filename, 'wb') as f:
            f.write(content)

    @staticmethod
    @contextlib.contextmanager
    def _main_file_creation(data, filename):
        with open('{}{}'.format(''.join(str(getattr(datetime.datetime.now(), i)) for i in ['day', 'month', 'year', 'minute', 'hour', 'second']), filename), 'a') as f:
            f.write(data)
        yield

    @staticmethod
    @contextlib.contextmanager
    def _write_binary(string_data, filename):
        with open('{}{}.bin'.format(''.join(str(getattr(datetime.datetime.now(), i)) for i in ['day', 'month', 'year', 'minute', 'hour', 'second']), re.sub('\.\w+$', '', filename)), 'wb') as f:
            f.write(string_data.encode('ascii'))
        yield

    @staticmethod
    def _test_bin_array(result):
        bin_array = array.array('B')
        for i in re.findall('[\w\W]{8}', result):
            bin_array.append(int(i[::-1]), 2)
        return bin_array

    @classmethod
    def print_structure(cls, d):
        _left, _parent, _right = d
        return '{}{}\n'.format(' '*len(_left), _parent)+'{}/  \\\n'.format(' '*len(_left))+"{}{}{}".format(cls.print_structure(_left) if not isinstance(_left, tuple) else _left, ' '*len(_left), cls.print_structure(_right) if not isinstance(_right, tuple) else _right)

    @staticmethod
    def get_counts(file_data):
        counts = Counter(file_data)
        return sorted([(i, counts.get(i)) for i in counts], key=lambda x:x[-1])




if __name__ == '__main__':
    @time_compression
    def compress():
        with Compress('simplerhobbit.txt') as c:
            pass

    @time_compression
    def decompress():
        with Decompress('35201885420simplerhobbit.bin') as c:
            pass

        print(('-'*10)+'compression finished'+('-'*10))
        print(size_comparison('simplerhobbit.txt', '35201885420simplerhobbit.bin'))

    def size_comparison(a, b):
        _a, _b = os.path.getsize(a), os.path.getsize(b)
        return '{}: {}, {}(compressed): {}. Total difference: {} bytes'.format(a, _a, b, _b, _a - _b)

    #print(size_comparison('largestexampleltr.txt', ))
    #print(size_comparison('largestexampleltr.txt', '152018301116largestexampleltr.txt'))
    def test_toBinary():
        import pickle
        for a, b in pickle.load(open('binary_checks.txt', 'rb')):
            assert Compress._to_Binary(int(a)) == b, '{} -> {}, got {}'.format(a, b, Compress._to_Binary(a))
        print('all test cases passed')

    decompress()
    #print(size_comparison('simplerhobbit.txt', '35201885420simplerhobbit.bin'))
    '''
    content, header = Decompress._master_header_Parser(open('252018411331lastltrchecktest1.txt').read())
    print('='*20)
    print(header)
    print(content)
    '''


    #print(list(filter(lambda x:len(x[1:]) > 1, [[[b]]+[h for _, h in d.items() if b.startswith(h) or h.startswith(b)] for _, b in d.items()])))
