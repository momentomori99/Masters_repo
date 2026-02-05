#### 1.2.112 
import numpy as np

def midpoint_rule(f,a,b,n):
    dx=(b-a)/n # lengde av intervaller
    xmid=np.arange(a+0.5*dx, b, dx ) # [a+0.5*dx, a+1.5*dx, ..., b-0.5*dx]
    fverdier=f(xmid)
    S= fverdier.sum()
    return S*dx

def trapezoidal_rule(f,a,b,n):
    dx=(b-a)/n
    x=np.linspace(a,b,n+1) #[a, a+dx, ..., b] (n+1 punkter)
    fverdier=f(x)
    S=0.5*(fverdier[0]+fverdier[-1])+np.sum(fverdier[1:-1])
    return S*dx
def simpsons_method(f,a,b,n):
    assert(n%2==0)
    dx=(b-a)/n
    x=np.linspace(a,b,n+1)
    xend=x[[0,-1]] # [x[0], x[n]]
    xodd=x[1:-1:2] # [x[1], x[3], ..., x[n-1]]
    xeven=x[2:-1:2] # [x[2], x[4], ..., x[n-2]]
    S= f(xend).sum()+4*f(xodd).sum()+2*f(xeven).sum()
    return S*dx/3


import sympy as sp
t = sp.symbols('t')
# Parametric curve
x = sp.exp(t) * sp.cos(t)
y = sp.exp(t) * sp.sin(t)

# Arc length integrand: sqrt((dx/dt)^2 + (dy/dt)^2)
dx = sp.diff(x, t)
dy = sp.diff(y, t)
integrand = sp.sqrt(dx**2 + dy**2)

# Interval
a = sp.Integer(0)
b = sp.pi/2

# Exact arc length (symbolic)
L_exact = sp.simplify(sp.integrate(integrand, (t, a, b)))

# Decimal rounded to 3 places
L_decimal = sp.N(L_exact, 20)          # high precision decimal
L_3dp = sp.N(L_exact, 6)               # enough precision for 3dp display

print("Integrand:", sp.simplify(integrand))
print("Exact L:", L_exact)
print("Decimal L:", L_decimal)
print("Rounded to 3dp:", float(L_decimal.evalf()))
print("3dp:", format(float(L_decimal), ".3f"))