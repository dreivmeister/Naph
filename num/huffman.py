import sys
import heapq
import string
import random
from dataclasses import dataclass, field
from typing import Any


class Node:
    def __init__(self, data=None, freq=None, left=None, right=None):
        self.data = data
        self.freq = freq
        self.left = left
        self.right = right

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

def calc_freqs(msg):
    symbols = list(set(msg))
    
    freqs = {}
    for sym in symbols:
        freqs[sym] = msg.count(sym)
    
    return freqs

def huffman(frequencies):
    # frequencies is dict of symbol - frequency pairs
    pq = []
    n = 0
    for char, freq in frequencies.items():
        n += 1
        heapq.heappush(pq, PrioritizedItem(freq, Node(data=char, freq=freq)))
    for _ in range(n-1):
        # two nodes with lowest frequencies
        x = heapq.heappop(pq).item
        y = heapq.heappop(pq).item
        
        z = Node(freq=x.freq+y.freq,left=x,right=y)
        # insert z into pq
        heapq.heappush(pq, PrioritizedItem(z.freq, z))
    root = heapq.heappop(pq).item
    return root

def calc_encodings(root, pre="", encodings={}):
    if root is None:
        return encodings
    
    # leaf node
    if root.data is not None:
        encodings[root.data] = pre
        
    # internal node
    calc_encodings(root.left, pre + '0', encodings)
    calc_encodings(root.right, pre + '1', encodings)
    return encodings

def encode_string(msg, encodings):
    enc_msg = ""
    
    for c in msg:
        enc_msg += encodings[c]
    
    return enc_msg

def decode_string(enc_msg, root):
    msg = ""
    
    node = root
    for c in enc_msg:
        if c == '0':
            node = node.left
        else:
            node = node.right
        if node.data is not None:
            msg += node.data
            node = root
    return msg
    
    


if __name__=="__main__":
    if len(sys.argv) > 1:
        msg = sys.argv[1]
        print(f"original msg: {msg}")
        
        freq_table = calc_freqs(msg)    
        huff_root = huffman(freq_table)
        enc_table = calc_encodings(huff_root)
        enc_msg = encode_string(msg, enc_table)
        
        print(f"encoded msg: {enc_msg}")
        print(f"number of bits in encoded msg: {len(enc_msg)}")
        
        #dec_msg = decode_string(enc_msg, huff_root)
        

    # testing    
    # N = 20
    # cnt = 0
    # for i in range(100):
    #     msg = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
    
    #     freq_table = calc_freqs(msg)    
        
    #     huff_root = huffman(freq_table)
        
    #     enc_table = calc_encodings(huff_root)
        
    #     enc_msg = encode_string(msg, enc_table)
        
    #     dec_msg = decode_string(enc_msg, huff_root)
        
    #     if dec_msg == msg:
    #         cnt += 1
    #     else:
    #         print('error')
    # print(f"perc: {cnt/100*100}")
    

    
    