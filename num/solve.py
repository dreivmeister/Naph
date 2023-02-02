def newtons_method(f,fp,x0,tol,N):
    i = 0
    fc = abs(f(x0))
    while fc > tol:
        xc = x0 - (f(x0)/fp(x0))
        fc = abs(f(xc))
        x0 = xc
        i += 1
        if i>N:
            print('Newtons method failed')
            return
    return x0

def damp_newtons_method(f,fp,x0,tol,N,lamb=1):
    i = 0
    fc = abs(f(x0))
    while fc > tol:
        xc = x0 - lamb*(f(x0)/fp(x0))
        lamb /= 2
        fc = abs(f(xc))
        x0 = xc
        i += 1
        if i>N:
            print('Newtons method failed')
            return
    return x0