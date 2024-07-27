#importing modules
import numpy as np
np.random.seed(12)
np.stack([1,6,3],[9,2,4],axis=2)
print(np.random.random(10))
#sample data
x= np.array([1,2,4,1,5])
y=np.array([3,2,4,1,2])
#cal mean of x,y
mean_x=np.mean(x)
mean_y=np.mean(y)
#cal num and deno
num=np.sum((x-mean_x)*(y-mean_y))
deno=np.sum((x-mean_x)**2)
#cal slope and intercept using num and deno
m=num/deno
b=mean_y-m*mean_x
#print slope and intercept
print("slope",m,"Intercept",b)
