# ASSUMPTIONS:
# no commas anywhere
# cant assign strings with spaces
# single argument functions
# all expressions on one line
# only print content of variables (no strings)

import math
import operator as op

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (int, float)     # A Scheme Number is implemented as a Python int or float
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = list             # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List
Env    = dict             # A Scheme environment (defined below) is a mapping of {variable: value}


filename = 'example_program.txt'

def tokenize(filename: str) -> list:    
    # each token is a line
    lines = []
    with open(filename) as file:
        for line in file:
            line = line.rstrip().lstrip()
            if line == '':
                continue
            if 'if' in line or 'while' in line or 'func' in line:
                lines.append(' ( ' + line)
            elif 'end' in line:
                lines.append(line + ' ) ')
            else:
                lines.append(' ( ' + line + ' ) ')

    tokenized = " ".join(lines).split()
    tokenized.insert(0, '(')
    tokenized.append(')')    
    return tokenized

#tokenized = tokenize(filename)

#print(tokenized)

def read_from_tokens(tokens: list) -> Exp:
    if len(tokens) == 0:
        return
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        return L
    elif token == ')':
        return
    else:
        return atom(token)

def atom(token: str) -> Atom:
    if token[-1] == ':':
        token = token[:-1]
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            token = str(token)
            if token[0] == "'" and token[-1] == "'":
                return token[1:-1]
            return token

def parse(program: str) -> Exp:
    return read_from_tokens(tokenize(program))

#parsed = parse(filename)

# for p in parsed:
#     print(p)
#print(parsed)


class Env(dict):
    #"An environment: a dict of {'var':val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        #"Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


def standard_env() -> Env:
    #"An environment with some Scheme standard procedures."
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '==':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_,
        'expt':    pow,
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: List(x), 
        'list?':   lambda x: isinstance(x, List), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),  
		'print':   print,
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env

global_env = standard_env()


def eval(x: Exp, env=global_env):
    # if its an Atom return (str, int, float)
    if isinstance(x, Atom):
        print(type(x))
        return x
    # if isinstance(x, Symbol):
    #     #print("{} : {} Symbol".format(x,env[x]))
    #     return x
    #     #return env[x]
    # elif isinstance(x, Number):
    #     print("{} : {} Number".format(x, type(x)))
    #     return x
    
    # num define
    elif x[0] == 'num':
        typ, symbol, _, exp = x
        env[symbol] = eval(exp, env)
    # str define
    elif x[0] == 'str':
        typ, symbol, _, exp = x
        env[symbol] = eval(exp, env)
    elif x[0] == 'print':
        # only variable contents can be printed
        symbol, exp = x
        try: print(env[exp])
        except KeyError:
            raise AttributeError('{} not defined'.format(exp))
    elif x[0] == 'if':
        _, ts, comp, te, conseq, _ = x
        exp = (conseq if eval("".join([ts,comp,str(te)]),env) else None)
        return eval(exp, env)
        
    
    
        
        
    
parsed = parse(filename)
#parsed.insert(0, 'begin')
print(parsed)

for p in parsed:
    print('-----------------------')
    print(p)
    res = eval(p, global_env)
    print(res)
print('-----------------------')
print(global_env['a'])



# def gen_end_token(index, lines):
#     token = []
#     while lines[index] != 'end':
#         token.append(lines[index])
#         index += 1
#     token.append(lines[index])
#     return token, index

# def gen_token(line):
#     i = 0
#     token = []
#     if lines[i].startswith('if') or lines[i].startswith('while'):
#         token, end_index = gen_end_token(i, lines)
#         i = end_index
#     elif lines[i].startswith('func'):
#         token, end_index = gen_end_token(i, lines)
#         i = end_index
#     elif lines[i].startswith('str'):
#         token = lines[i].split()
#         equals_index = token.index('=')
#         token[equals_index+1:] = [' '.join(token[equals_index+1:])[1:-1]]
#     elif lines[i].startswith('num'):
#         token = lines[i].split()
#     elif lines[i].startswith('print'):
#         # can only print content of variables
#         token = lines[i].split('(')
#         # remove closing parentheses
#         token[-1] = token[-1][:-1]
#     i += 1
#     return token


# tokens = []
# while len(lines) > 0:
#     tokens.append(gen_token(lines))
#     lines.pop(0)
    
    
# for t in tokens:
#     print(t)    

#print(tokens)




# tokens = []
# i = 0
# while i < len(lines):
#     token, i = gen_token(i, lines)
#     tokens.append(token)
    
#     print(token)










# for j, l in enumerate(list_of_tokens):
#     # assign variable str
#     if type(l) == list:
#         continue
#     if l.startswith('if'):
#         # l = l.split()
#         if_index = None
#         end_index = None

#         for i, elem in enumerate(list_of_tokens):
#             if type(elem) == list:
#                 continue
#             if elem.startswith('if'):
#                 if_index = i
#             elif elem.startswith('end'):
#                 end_index = i
#                 break
#         if if_index is not None and end_index is not None:
#             list_of_tokens[if_index:end_index+1] = [list_of_tokens[if_index:end_index+1]]
#         list_of_tokens.append(l)
#     elif l.startswith('str'):
#         l = l.split()
#         equals_index = l.index('=')
#         l[equals_index+1:] = [' '.join(l[equals_index+1:])[1:-1]]
#         list_of_tokens.append(l)
#     # assign variable num
#     elif l.startswith('num'):
#         list_of_tokens.append(l.split())
#     elif l.startswith('print'):
#         # can only print content of variables
#         l = l.split('(')
#         # remove closing parentheses
#         l[-1] = l[-1][:-1]
#         list_of_tokens.append(l)
#     elif l.startswith('while'):
#         list_of_tokens.append(l)
#     elif l.startswith('func'):
#         list_of_tokens.append(l)
#     else:
#         list_of_tokens.append(l)
# #print(list_of_tokens)




# print console output
# if: ... end
# while: ... end
# func(...): ... end