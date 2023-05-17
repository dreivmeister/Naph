import math
import matplotlib.pyplot as plt

# always holds two representations
class Complex:
    def __init__(self, real=None, imag=None, r=None, phi=None) -> None:
        # it expects that representations are fully given 
        # either both real and imag or both r and phi or all four
        # rest is undefined behaviour
        
        #standard repr
        self.real = real
        self.imag = imag
        
        #polar repr
        self.r = r
        if phi is not None:
            self.phi = phi
            self.cos_phi = math.cos(phi)
            self.sin_phi = math.sin(phi)
        else:
            self.phi = phi
            self.cos_phi = None
            self.sin_phi = None
            
        # check which repr to calculate internally
        # calc standard from polar
        if self.real is None and self.imag is None and self.r is not None and self.phi is not None:
            self.real = self.r * self.cos_phi
            self.imag = self.r * self.sin_phi
        # calc polar from standard
        if self.real is not None and self.imag is not None and self.r is None and self.phi is None:
            self.r = math.sqrt(self.real**2+self.imag**2)
            # is not zero
            assert abs(self.r - 0.0) > 1e-10 and abs(self.real - 0.0) > 1e-10 
            self.phi = math.atan2(self.imag,self.real)
            self.cos_phi = self.real/self.r
            self.sin_phi = self.imag/self.r
        
        self.abs = math.sqrt(self.real**2+self.imag**2)
            
    
    def get_real(self):
        return self.real
    def get_imag(self):
        return self.imag
    def get_conjugate(self):
        d = self.real**2+self.imag**2
        return Complex(self.real/d,-self.imag/d)
    # overloads
    def __repr__(self) -> str:
        if self.r is not None:
            return f"({str(self.real)}, {str(self.imag)})\n({str(self.r)}, {str(self.cos_phi)}, {str(self.sin_phi)})"
        return f"({str(self.real)}, {str(self.imag)})"
    def __neg__(self):
        return Complex(-self.real,-self.other)
    def __add__(self, other):
        return Complex(self.real+other.real,self.imag+other.imag)
    def __sub__(self, other):
        return self + (-other)
    def __mul__(self, other):
        return Complex(self.real*other.real-self.imag*other.imag,
                       self.real*other.imag+self.imag*other.real)
    def __truediv__(self, other):
        d = other.real**2+other.imag**2
        return Complex((self.real*other.real+self.imag*other.imag)/d,(self.imag*other.real-self.real*other.imag)/d)
    def __pow__(self, other):
        return Complex(r=self.r**other, phi=self.phi*other)


def plot_numbers(numbers, type='gaussian'):
    if type == 'gaussian':
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        for z in numbers:
            plt.plot([0,z.real],[0,z.imag],'ro-',label='python')
        limit = math.ceil(max([max(i.real,i.imag) for i in numbers]))+2 # set limits for axis
        plt.xlim((-limit,limit))
        plt.ylim((-limit,limit))
        plt.show()
    elif type == 'polar':
        # plot number in polar coordinates
        print('polar plot will be added later, come back later')
        return

    
    

#z2 = Complex(3.0,4.0)


#z1.plot_number()

# print(z1 + z2)
# print(z1 * z2)
# print(z1 / z2)
# print(abs(z1))

# z1.calculate_polar()

# print(Complex(4.0,-6.0) / Complex(-1.0,-2.0))
    

if __name__=="__main__":
    z1 = Complex(1.0,2.0)
    z2 = Complex(4.0,1.0)
    z3 = Complex(3.0,8.0)
    
    
    print(z1 + z2)
    
    plot_numbers([z1,z2,z3])