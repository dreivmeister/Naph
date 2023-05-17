# IMPORTS
import math






# CONSTANTS
p = 307 #3
q = 859 # 11


# GENERAL FUNCTIONS
def euler_totient(p, q):
    # phi(N)
    return (p-1)*(q-1)

def is_relatively_prime(N, t):
    gcd = math.gcd(N, t)
    return gcd == 1

# BREAK RSA
def factorize(N):
    factors = []
    d = 2
    while d * d <= N:
        while (N % d) == 0:
            factors.append(d)
            N //= d
        d += 1
    if N > 1:
        factors.append(N)
    return factors



# ASSUMING LOWERCASE ONLY
def get_alphabet_position(letter):
    if letter == ' ':
        return 0
    if letter.isupper():
        return ord(letter) - ord('A') + 1
    return ord(letter) - ord('a') + 1

def get_letter_from_alphabet_position(position):
    if position == 0:
        return ' '
    if position > 26:
        return chr(position-26 + ord('A') - 1)
    return chr(position + ord('a') - 1)



# ENCRYPTION
def compute_t(N, phi_N):
    l = [i for i in range(2, phi_N+1)]
    
    for t in l:
        if is_relatively_prime(N, t) and is_relatively_prime(phi_N, t):
            return t
        
def convert_msg_forward(msg, encryption_lock):
    l = []
    for m in msg:
        position = get_alphabet_position(m)
        remainder = position**encryption_lock[0] % encryption_lock[1]
        l.append(remainder)
    return l


def encrypt(p, q):
    N = p * q
    phi_N = euler_totient(p, q)
    t = compute_t(N, phi_N)
    encryption_lock = (t, N)
    return encryption_lock, phi_N
    


# DECRYPTION
def compute_d(phi_N, t):
    l = [i for i in range(1,phi_N+1)]
    
    for d in l:
        if t*d % phi_N == 1:
            return d
    return None

def convert_msg_backward(msg, decryption_key):
    l = []
    for m in msg:
        remainder = m**decryption_key[0] % decryption_key[1]
        l.append(get_letter_from_alphabet_position(remainder))
    return "".join(l)


def decrypt(p, q, phi_N, t):
    N = p * q
    d = compute_d(phi_N, t)
    if d is None:
        return False
    decryption_key = (d, N)
    return decryption_key
    





if __name__=='__main__':
    #print(get_letter_from_alphabet_position(29))
    msg = 'hello world'
    print(msg)
    encrpytion_lock, phi_N = encrypt(p, q)
    #print(encrpytion_lock)
    encrypted_msg = convert_msg_forward(msg, encrpytion_lock)
    #print(encrypted_msg)
    
    decryption_key = decrypt(p, q, phi_N, encrpytion_lock[0])
    #print(decryption_key)
    decrypted_msg = convert_msg_backward(encrypted_msg, decryption_key)
    print(decrypted_msg)
    
    
    # HOW TO BREAK IT
    # N = encrpytion_lock[1]
    # factors = factorize(N)
    # print(factors)
    
    
