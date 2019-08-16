import hashlib
import struct


class MD5(object):
    def __init__(self):
        self.m = hashlib.md5()

    def update(self, o):
        if isinstance(o, str):
            self.m.update(o.encode('utf8', errors='backslashreplace'))
        elif isinstance(o, int):
            self.m.update(str(o).encode('utf8'))
        elif isinstance(o, float):
            self.m.update(struct.pack('<f', o))
        elif o is None:
            pass
        else:
            raise TypeError('Unsupported hash object: %s' % type(o))

    def __call__(self, *args):
        for arg in args:
            self.update(arg)
        return self

    def hexdigest(self):
        return self.m.hexdigest()
