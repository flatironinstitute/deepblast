# Author : Tristian Bepler

import numpy as np


class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        z = self.encoding[x]
        return z

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)


DNA = Alphabet(b'ACGT')


class Uniprot21(Alphabet):
    def __init__(self, mask=False):
        chars = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(
            chars, encoding=encoding, mask=mask, missing=20)


class UniprotTokenizer:

    def __init__(self, pad_ends=False):
        self.alphabet = Uniprot21()
        self.pad_ends = pad_ends

    def __call__(self, x):
        # encode sequence
        s = np.array(x.upper())
        x = self.alphabet.encode(s)
        # pad with start/stop token
        if self.pad_ends:
            z = np.zeros(len(x) + 2, dtype=x.dtype)
            z[1:-1] = x
            z[0] = 20   # pad start
            z[-1] = 20  # pad end
            return z
        else:
            return x
