python39.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def kernal(point,xmat,k):
m,n=np.shape(xmat)
weights=np.mat(np.eye((m)))
for j in range(m):
diff=point-x[j]
weights[j,j]=np.exp(diff*diff.T/(-2.0*k**2))
return weights
def localweight(point,xmat,ymat,k):
wt=kernal(point,xmat,k)
w=(x.T*(wt*x)).I*(x.T*wt*ymat.T)
return w
def localweightregression(xmat,ymat,k):
m,n=np.shape(xmat)
ypred=np.zeros(m)
#print(m)
#print(n)
#print(ypred)
for i in range(m):
ypred[i]=xmat[i]*localweight(xmat[i],xmat,ymat,k)
print(ypred[i])
return ypred
data=pd.read_csv('Tips.csv')
cola=np.array(data.total_bill)
colb=np.array(data.tip)
#print(cola)
#print(colb)
mcola=np.mat(cola)
#print(mcola)
mcolb=np.mat(colb)
#print(mcolb)
m=np.shape(mcolb)[1]
#print(m)
one=np.ones((1,m),dtype=int)
#print(one)
x=np.hstack((one.T,mcola.T))
print(x.shape)
#print(x)
ypred=localweightregression(x,mcolb,0.5)
#print(ypred)
xsort=x.copy()
xsort.sort(axis=0)
#print(xsort)
plt.scatter(cola,colb,color='blue')
plt.plot(xsort[:,1],ypred[x[:,1].argsort(0)],color='yellow',linew
idth=5)
plt.xlabel('Total Bill')
plt.ylabel('tip')
plt.show()