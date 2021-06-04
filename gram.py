import numpy 
import math
def proj(u,v,k):
    uk=numpy.dot(k,u)
    numerator=numpy.dot(v,uk)
    denom=numpy.dot(u,uk)
    output=u*(numerator/denom)
    return output

def normvec(vec,k):
    fac = numpy.einsum('i,ij,j',vec,k,vec)
    vec /= math.sqrt(fac)
    return vec

def schmidt(input,k,states,ndr):
    output=numpy.zeros((states,ndr))
    output[0]=input[0]
    for i in range(1,states):
        temp=numpy.zeros(ndr)
        for j in range(i):
            temp=temp+proj(output[j],input[i],k)
        output[i]=input[i]-temp
    for i in range(states):
        output[i]=normvec(output[i],k)
    return output





