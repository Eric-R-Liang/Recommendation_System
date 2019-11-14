from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd

A = array([[1,2],[4,5],[7,8]])
print(A)
U,s,VT = svd(A)
print(U)
print(s)
print(VT)
Sigma = zeros((A.shape[0],A.shape[1]))
Sigma[:A.shape[1],:A.shape[1]] = diag(s)
B = U.dot(Sigma.dot(VT))
