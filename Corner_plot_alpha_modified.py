#Metropolis Algortihm in Log Space - Success
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import corner

#The following few lines define the main straight line we will be using for all our purposes.
x_array = np.arange(20.0)
m_true = 2; c_true = 7
sigma_true = 0.5
y_array = m_true*x_array + c_true
sigma_c,sigma_alpha = 0.1,0.01
# The following line generates random points from a gaussian distribution arouond every data point
y_data = np.random.normal(y_array,sigma_true,len(x_array))
corner_data_alpha = np.zeros(2000);
corner_data_c = np.zeros(2000);

# Chi-Squared Minimization, defining a sentinel value
# We need to keep in mind that the chi-squared sum we are going to minimize is faulty,
# as we will be maximizing only the likelihood probability. We need to maximise the prior probability
# I will be assuming a completely flat prior in this particular plot.
#This function computes the prior probability, given the values of the free parameters, in this case m and c. 
def prior_p(alpha,c):
    if(alpha >= 0 and alpha <= np.pi and c >= 0 and c <= 10):
        return 0;
    else:
        return (-np.inf);

def likelihood_p( y_current ):   
    sum_current = np.sum((y_current - y_array)*(y_current - y_array)/(2*sigma_true*sigma_true))
    return (-sum_current)

def likelihood_adap_p( y_current,y_mean ):
    sum_current = np.sum((y_current - y_mean)*(y_current - y_mean))
    return (-sum_current)
#First Chain 
length = 1000
count =1;
alpha_current = np.random.uniform(0.0,3.1415/2)
m_current = np.tan(alpha_current)
c_current = np.random.uniform(0.0,10.0)
y_current = m_current*x_array + c_current
p_current = likelihood_p(y_current) + prior_p(alpha_current,c_current)

#We will be plotting two graphs, one for the variation of m and the other for the variation with c.
#We have to run a certain number of chains, thus we expect the m and c values for all of them to converge


alpha_array1 = np.zeros(length)
c_array1 = np.zeros(length)
alpha_array1[count-1] = alpha_current
c_array1[count-1] = c_current

while (count<length):
    alpha_proposed = np.random.normal(alpha_current,sigma_alpha)
    m_proposed = np.tan(alpha_proposed)
    c_proposed = np.random.normal(c_current,sigma_c)
    y_proposed = m_proposed*x_array + c_proposed
    p_proposed = likelihood_p(y_proposed) + prior_p(alpha_proposed,c_proposed);
    if(p_proposed > p_current):
        m_current = m_proposed
        c_current = c_proposed
        p_current = p_proposed
        alpha_current = alpha_proposed 
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)): #Make this -ve if necessary
            m_current = m_proposed
            c_current = c_proposed
            p_current = p_proposed
            alpha_current = alpha_proposed 
    count = count+1;
    alpha_array1[count-1]=alpha_current
    c_array1[count-1]=c_current
y_test1 = m_current*x_array + c_current;
corner_data_alpha[0:500]=alpha_array1[500:1000]
corner_data_c[0:500]=c_array1[500:1000]
print("m=")
print(m_current)
print("c=")
print(c_current)
alpha_start1= alpha_current
c_start1 = c_current
#______________________________________________________________________

#Second Chain

count =1;
alpha_current = np.random.uniform(0.0,3.1415/2)
m_current = np.tan(alpha_current)
c_current = np.random.uniform(0.0,10.0)
y_current = m_current*x_array + c_current
p_current = likelihood_p(y_current) + prior_p(alpha_current,c_current)

#We will be plotting two graphs, one for the variation of m and the other for the variation with c.
#We have to run a certain number of chains, thus we expect the m and c values for all of them to converge


alpha_array2 = np.zeros(length)
c_array2 = np.zeros(length)
alpha_array2[count-1]=m_current
c_array2[count-1]=c_current

while (count<length):
    alpha_proposed = np.random.normal(alpha_current,sigma_alpha)
    m_proposed = np.tan(alpha_proposed)
    c_proposed = np.random.normal(c_current,sigma_c)
    y_proposed = m_proposed*x_array + c_proposed
    p_proposed = likelihood_p(y_proposed)  + prior_p(alpha_proposed,c_proposed)
    if(p_proposed > p_current):
        m_current = m_proposed
        c_current = c_proposed
        p_current = p_proposed
        alpha_current = alpha_proposed 
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            m_current = m_proposed
            c_current = c_proposed
            p_current = p_proposed
            alpha_current = alpha_proposed 
    count = count+1;
    alpha_array2[count-1]=alpha_current
    c_array2[count-1]=c_current
#_________________________________________________________________________
print("m=")
print(m_current)
print("c=")
print(c_current)
corner_data_alpha[500:1000]=alpha_array2[500:1000]
corner_data_c[500:1000]=c_array2[500:1000]
alpha_start2 = alpha_current
c_start2 = c_current
y_test2 = m_current*x_array + c_current;

#Third Chain

count =1;
alpha_current = np.random.uniform(0.0,3.1415/2)
m_current = np.tan(alpha_current)
c_current = np.random.uniform(0.0,10.0)
y_current = m_current*x_array + c_current
p_current = likelihood_p(y_current) + prior_p(alpha_current,c_current)

#We will be plotting two graphs, one for the variation of m and the other for the variation with c.
#We have to run a certain number of chains, thus we expect the m and c values for all of them to converge


alpha_array3 = np.zeros(length)
c_array3 = np.zeros(length)
alpha_array3[count-1]=alpha_current
c_array3[count-1]=c_current

while (count<length):
    alpha_proposed = np.random.normal(alpha_current,sigma_alpha)
    m_proposed = np.tan(alpha_proposed)
    c_proposed = np.random.normal(c_current,sigma_c)
    y_proposed = m_proposed*x_array + c_proposed
    p_proposed = likelihood_p(y_proposed)  + prior_p(alpha_proposed,c_proposed)
    if(p_proposed > p_current):
        m_current = m_proposed
        c_current = c_proposed
        p_current = p_proposed
        alpha_current = alpha_proposed 
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            m_current = m_proposed
            c_current = c_proposed
            p_current = p_proposed
            alpha_current = alpha_proposed 
    count = count+1;
    alpha_array3[count-1]=alpha_current
    c_array3[count-1]=c_current
#______________________________________________________________________

corner_data_alpha[1000:1500]=alpha_array3[500:1000]
corner_data_c[1000:1500]=c_array3[500:1000]
alpha_start3 = alpha_current
c_start3 = c_current
y_test3 = m_current*x_array + c_current;
print("m=")
print(m_current)
print("c=")
print(c_current)

#Fourth Chain

count =1;
alpha_current = np.random.uniform(0.0,3.1415/2)
m_current = np.tan(alpha_current)
c_current = np.random.uniform(0.0,10.0)
y_current = m_current*x_array + c_current
p_current = likelihood_p(y_current) + prior_p(alpha_current,c_current)

#We will be plotting two graphs, one for the variation of m and the other for the variation with c.
#We have to run a certain number of chains, thus we expect the m and c values for all of them to converge


alpha_array4 = np.zeros(length)
c_array4 = np.zeros(length)
alpha_array4[count-1]=alpha_current
c_array4[count-1]=c_current

while (count<length):
    alpha_proposed = np.random.normal(alpha_current,sigma_alpha)
    m_proposed = np.tan(alpha_proposed)
    c_proposed = np.random.normal(c_current,sigma_c)
    y_proposed = m_proposed*x_array + c_proposed
    p_proposed = likelihood_p(y_proposed) + prior_p(alpha_proposed,c_proposed)
    if(p_proposed > p_current):
        m_current = m_proposed
        c_current = c_proposed
        p_current = p_proposed
        alpha_current = alpha_proposed 
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            m_current = m_proposed
            c_current = c_proposed
            p_current = p_proposed
            alpha_current = alpha_proposed 
    count = count+1;
    alpha_array4[count-1]=alpha_current
    c_array4[count-1]=c_current
#___________________________________________________________________________________

corner_data_alpha[1500:2000]=alpha_array4[500:1000]
corner_data_c[1500:2000]=c_array4[500:1000]
alpha_start4 = alpha_current
c_start4 = c_current
y_test4 = m_current*x_array + c_current;
print("m=")
print(m_current)
print("c=")
print(c_current)


corner2_data_alpha = np.zeros(2000)
corner2_data_c = np.zeros(2000)
covariance_mat = np.cov(np.stack((corner_data_alpha,corner_data_c)))
print("Covariance = ")
print(covariance_mat)
alpha_points1 = np.zeros(1000)
c_points1 = np.zeros(1000)
#First Chain ________________________________________________________________________
y_mean = np.tan(alpha_start1)*x_array + c_start1
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start1,c_start1]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha_points1[count-1] = alpha_c_current[0]
c_points1[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha_points1[count-1] = alpha_c_current[0]
    c_points1[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner2_data_alpha[0:500]=alpha_points1[500:1000]
corner2_data_c[0:500]=c_points1[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])
alpha_start1 = alpha_c_current[0]
c_start1 = alpha_c_current[1]


alpha_points2 = np.zeros(1000)
c_points2 = np.zeros(1000)
#Second Chain ________________________________________________________________________
y_mean = np.tan(alpha_start2)*x_array + c_start2
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start2,c_start2]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha_points2[count-1] = alpha_c_current[0]
c_points2[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha_points2[count-1] = alpha_c_current[0]
    c_points2[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner2_data_alpha[500:1000]=alpha_points2[500:1000]
corner2_data_c[500:1000]=c_points2[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])
alpha_start2 = alpha_c_current[0]
c_start2 = alpha_c_current[1]

alpha_points3 = np.zeros(1000)
c_points3 = np.zeros(1000)
#Third Chain ________________________________________________________________________
y_mean = np.tan(alpha_start3)*x_array + c_start3
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start3,c_start3]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha_points3[count-1] = alpha_c_current[0]
c_points3[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha_points3[count-1] = alpha_c_current[0]
    c_points3[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner2_data_alpha[1000:1500]=alpha_points3[500:1000]
corner2_data_c[1000:1500]=c_points3[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])
alpha_start3 = alpha_c_current[0]
c_start3 = alpha_c_current[1]


alpha_points4 = np.zeros(1000)
c_points4 = np.zeros(1000)
#Fourth Chain ________________________________________________________________________
y_mean = np.tan(alpha_start4)*x_array + c_start4
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start4,c_start4]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha_points4[count-1] = alpha_c_current[0]
c_points4[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha_points4[count-1] = alpha_c_current[0]
    c_points4[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner2_data_alpha[1500:2000]=alpha_points4[500:1000]
corner2_data_c[1500:2000]=c_points4[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])
alpha_start4 = alpha_c_current[0]
c_start4 = alpha_c_current[1]

#________________________________________________________________
corner3_data_alpha = np.zeros(2000)
corner3_data_c = np.zeros(2000)
covariance_mat = np.cov(np.stack((corner2_data_alpha,corner2_data_c)))
print("Covariance = ")
print(covariance_mat)
alpha2_points1 = np.zeros(1000)
c2_points1 = np.zeros(1000)
#First Chain ________________________________________________________________________
y_mean = np.tan(alpha_start1)*x_array + c_start1
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start1,c_start1]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha2_points1[count-1] = alpha_c_current[0]
c2_points1[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha2_points1[count-1] = alpha_c_current[0]
    c2_points1[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner3_data_alpha[0:500]=alpha2_points1[500:1000]
corner3_data_c[0:500]=c2_points1[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])

alpha2_points2 = np.zeros(1000)
c2_points2 = np.zeros(1000)
#Second Chain ________________________________________________________________________
y_mean = np.tan(alpha_start2)*x_array + c_start2
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start2,c_start2]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha2_points2[count-1] = alpha_c_current[0]
c2_points2[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha2_points2[count-1] = alpha_c_current[0]
    c2_points2[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner3_data_alpha[500:1000]=alpha2_points2[500:1000]
corner3_data_c[500:1000]=c2_points2[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])

alpha2_points3 = np.zeros(1000)
c2_points3 = np.zeros(1000)
#Third Chain ________________________________________________________________________
y_mean = np.tan(alpha_start3)*x_array + c_start3
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start3,c_start3]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha2_points3[count-1] = alpha_c_current[0]
c2_points3[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha2_points3[count-1] = alpha_c_current[0]
    c2_points3[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner3_data_alpha[1000:1500]=alpha2_points3[500:1000]
corner3_data_c[1000:1500]=c2_points3[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])

alpha2_points4 = np.zeros(1000)
c2_points4 = np.zeros(1000)
#Fourth Chain ________________________________________________________________________
y_mean = np.tan(alpha_start4)*x_array + c_start4
alpha_c_current = np.random.multivariate_normal(np.array([alpha_start4,c_start4]),covariance_mat)
y_current = np.tan(alpha_c_current[0])*x_array + alpha_c_current[1]
alpha2_points4[count-1] = alpha_c_current[0]
c2_points4[count-1] = alpha_c_current[1]
p_current = likelihood_adap_p(y_current,y_mean) + prior_p(alpha_c_current[0],alpha_c_current[1])
count = 1;
length = 1000
while(count < length):
    alpha_c_proposed = np.random.multivariate_normal(alpha_c_current,covariance_mat)
    y_proposed = np.tan(alpha_c_proposed[0])*x_array + alpha_c_proposed[1]
    p_proposed = likelihood_adap_p(y_proposed,y_mean) + prior_p(alpha_c_proposed[0],alpha_c_proposed[1])
    if(p_proposed > p_current):
        alpha_c_current = alpha_c_proposed
        p_current = p_proposed           
    elif(np.exp(p_current) != 0):
        p_transition = p_proposed - p_current
        p_random = np.random.uniform();
        if(p_random <= np.exp(p_transition)):
            alpha_c_current = alpha_c_proposed
            p_current = p_proposed 
    count = count+1;
    alpha2_points4[count-1] = alpha_c_current[0]
    c2_points4[count-1] = alpha_c_current[1]

#________________________________________________________________________
corner3_data_alpha[1500:2000]=alpha2_points4[500:1000]
corner3_data_c[1500:2000]=c2_points4[500:1000]
print("m=")
print(np.tan(alpha_c_current[0]))
print("c=")
print(alpha_c_current[1])

#Finding the values with max likelihood
alpha_max = corner2_data_alpha[0]
c_max = corner2_data_c[0]
y_max = np.tan(alpha_max)*x_array + c_max
p_max = likelihood_p(y_max) + prior_p(alpha_max,c_max)
for i in range(1,2000):
    alpha_current = corner2_data_alpha[i]
    c_current = corner2_data_c[i]
    y_current = np.tan(alpha_current)*x_array + c_current
    p_current = likelihood_p(y_max) + prior_p(alpha_current,c_current)
    if(p_current>p_max):
        p_max=p_current
        y_max=y_current
        alpha_max=alpha_current
        c_max=c_current

print("Final Values:")
print("m = ")
print(np.tan(alpha_max))
print(" c =")
print(c_max)

#Finding the values with max likelihood
alpha_max = corner3_data_alpha[0]
c_max = corner3_data_c[0]
y_max = np.tan(alpha_max)*x_array + c_max
p_max = likelihood_p(y_max) + prior_p(alpha_max,c_max)
for i in range(1,2000):
    alpha_current = corner3_data_alpha[i]
    c_current = corner3_data_c[i]
    y_current = np.tan(alpha_current)*x_array + c_current
    p_current = likelihood_p(y_max) + prior_p(alpha_current,c_current)
    if(p_current>p_max):
        p_max=p_current
        y_max=y_current
        alpha_max=alpha_current
        c_max=c_current

print("Final Values:")
print("m = ")
print(np.tan(alpha_max))
print(" c =")
print(c_max)


fig=plt.figure(1)
ax = fig.add_subplot(111)
ax.set_ylabel("Y")
ax.set_xlabel("X")
plt.plot(x_array,y_data,'o')
plt.errorbar(x_array,y_data,yerr=sigma_true,linestyle="none")
plt.plot(x_array,np.tan(alpha_max)*x_array + c_max)


levels = 1.0 - np.exp(-0.5*np.arange(1.0,2.1,0.5)**2)

data = np.column_stack((corner_data_alpha,corner_data_c))
plt.figure(2)
plt.plot(np.arange(length),np.tan(alpha_array1));
plt.plot(np.arange(length),np.tan(alpha_array2));
plt.plot(np.arange(length),np.tan(alpha_array3));
plt.plot(np.arange(length),np.tan(alpha_array4));
plt.figure(3)
plt.plot(np.arange(length),c_array1);
plt.plot(np.arange(length),c_array2);
plt.plot(np.arange(length),c_array3);
plt.plot(np.arange(length),c_array4);
plt.figure(4)
plt.plot(np.arange(length),np.tan(alpha_points1))
plt.plot(np.arange(length),np.tan(alpha_points2))
plt.plot(np.arange(length),np.tan(alpha_points3))
plt.plot(np.arange(length),np.tan(alpha_points4))
plt.figure(5)
plt.plot(np.arange(length),c_points1)
plt.plot(np.arange(length),c_points2)
plt.plot(np.arange(length),c_points3)
plt.plot(np.arange(length),c_points4)
plt.figure(6)
plt.plot(np.arange(length),np.tan(alpha2_points1))
plt.plot(np.arange(length),np.tan(alpha2_points2))
plt.plot(np.arange(length),np.tan(alpha2_points3))
plt.plot(np.arange(length),np.tan(alpha2_points4))
plt.figure(7)
plt.plot(np.arange(length),c2_points1)
plt.plot(np.arange(length),c2_points2)
plt.plot(np.arange(length),c2_points3)
plt.plot(np.arange(length),c2_points4)
figure1 = corner.corner(data, bins = 25, levels = levels, smooth = 1.0, smooth1d = 1.0)
data2 = np.column_stack((corner2_data_alpha,corner2_data_c))
data3 = np.column_stack((np.tan(corner3_data_alpha),corner3_data_c))
figure2 = corner.corner(data2, bins = 25, levels = levels, smooth = 1.5, smooth1d = 1.0)
figure3 = corner.corner(data3, bins = 35, levels = levels, smooth = 1.5, smooth1d = 1.0,show_titles = True,\
                        label_kwargs = dict(size = '20'),range = [[1.85,2.15],[5,10]],title_kwargs = dict(size='20'),fill_contours = True,\
                        quantiles = [0.158656,0.841344],labels = ["m","c"],truths = [np.tan(alpha_max),c_max])
plt.show()

