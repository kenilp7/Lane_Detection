import numpy as np
import matplotlib.pyplot as plt

def draw(x,y): # for plotting the line
    ln=plt.plot(x,y)
    plt.pause(0.0001)
    ln[0].remove()

def sigmoid(score): # find the probabilities using sigmoid function
    return 1/(1+np.exp(-score))

def calculate_error(line_para,points,y): # calculating the cross_entropy
    n=points.shape[0]
    p=sigmoid(points*line_para)
    cross_entropy=-(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))
    return cross_entropy

def gradient_descent(line_parameters,points,y,alpha):
    n=points.shape[0]
    for i in range(2000):
        p=sigmoid(points*line_parameters)
        gradient=points.T*(p-y)*(alpha/n)
        line_parameters=line_parameters-gradient # line parameters are updated after every iteration
        w1=line_parameters.item(0)
        w2=line_parameters.item(1)
        b=line_parameters.item(2)
        x1=np.array([points[:,0].min(),points[:,0].max()])
        x2=-b/w2 + (x1*(-w1/w2)) 
        draw(x1,x2)


n_pts=100
np.random.seed(0) # fix the random values

x_top=np.random.normal(10,2,n_pts) # normal dist. (center_pt,std_dev,n_pts)
y_top=np.random.normal(12,2,n_pts) # normal dist. (center_pt,std_dev,n_pts)

top_region=np.array([x_top,y_top]).T # points in top region (x,y)

x_btm=np.random.normal(5,2,n_pts) # normal dist. (center_pt,std_dev,n_pts)
y_btm=np.random.normal(6,2,n_pts) # normal dist. (center_pt,std_dev,n_pts)

btm_region=np.array([x_btm,y_btm]).T # points in bottom region (x,y)

bias=np.ones(2*n_pts) # ones array for multiplying it with bias value - 2* because of the vstacking in all_pts

all_pts=np.vstack((top_region,btm_region))
all_pts=np.c_[all_pts,bias] # adding bias array of ones to all points for z=WX+b multiplication

y=np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts*2,1) # the ground truth values (or labels) 0 or 1 in this case

line_para=np.matrix([np.zeros(3)]).T
_,ax=plt.subplots(figsize=(7,7)) 
ax.scatter(top_region[:,0],top_region[:,1],color='r') 
ax.scatter(btm_region[:,0],btm_region[:,1],color='b') 
gradient_descent(line_para,all_pts,y,0.06)
plt.show()