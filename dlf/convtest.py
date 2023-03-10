import numpy as np



def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    return [
        np.dot(
        signal[max(0,i):min(i+len(kernel),len(signal))],
        kernel[max(-i,0):len(signal)-i*(len(signal)-len(kernel)<i)])
        for i in range(1-len(kernel),len(signal))
    ]
    

    

print(convolve_1d([1, 1, 2, 2, 1], [1, 1, 1, 3]))