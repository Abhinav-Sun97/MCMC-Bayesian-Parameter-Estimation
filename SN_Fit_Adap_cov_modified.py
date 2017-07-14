# MCMC Technique to fit a curve through available supernovae data
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import scipy.special as sc
import corner

#The data points are stored in the 'jla_mub_0.txt' file
z=np.zeros(31); distance_moduli_data=np.zeros(31)
file = open("jla_mub_0.txt",'r')
count =0
for line in file:
    z[count], distance_moduli_data[count] = line.split(' ', 1)
    count = count +1
corner_data_omega_m = np.zeros(2000);
corner_data_h = np.zeros(2000);


#We have 31 data points, and hence we have a corresponding 31 X 31 Covariance matrix
cov_matrix=np.zeros(31*31); 
with open('jla_mub_covmatrix.txt') as f:
  cov_matrix = [float(i) for i in f]
cov_matrix=np.reshape(cov_matrix,(31,31))
sigma_h = 0.025
sigma_omega_m = 0.025
#We need to calculate the theoretical values as well, hence we need a function for that
distance_moduli_theo = np.zeros(31)
def distance_moduli_fn(z,h,omega_m):
    a = 1.0/(1+z)
    luminosity_distance = 3000/(1*a*a*np.sqrt(1-omega_m))*(-a*sc.hyp2f1(1/3,1/2,4/3,-omega_m/(1 - omega_m)) + sc.hyp2f1(1/3,1/2,4/3,-omega_m/(a*a*a*(1-omega_m))))
    try:
        distance_moduli = 25 - 5*np.log10(h)+5*np.log10(luminosity_distance)
    except:
        print(h)
        print(luminosity_distance)
        a=raw_input()
    return distance_moduli;
def likelihood(distance_moduli):
    p=multivariate_normal.pdf(distance_moduli,distance_moduli_data,cov_matrix)
    if(p!=0):
        return np.log(p);
    else:
        return (-np.inf);
def likelihood_adap_cov(distance_moduli,distance_moduli_mean):
    p=multivariate_normal.pdf(distance_moduli,distance_moduli_mean,cov_matrix)
    if(p!=0):
        return np.log(p);
    else:
        return (-np.inf);

def prior(h,omega_m):
    if(h>=0 and h<=1 and omega_m>=0 and omega_m<=1):
        return 0;
    else:
        return (-np.inf);

#We have now defined all the required functions. It is now time to run the chain(s).

# Chain -1
length = 2000
count =1;
h_current = np.random.uniform()
omega_m_current = np.random.uniform()
distance_moduli_theo_c = distance_moduli_fn(z,h_current,omega_m_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current)
h_values_chain1 = np.zeros(length)
omega_m_values_chain1 = np.zeros(length)
h_values_chain1[count-1] = h_current
omega_m_values_chain1[count-1] = omega_m_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1):
        distance_moduli_theo_p = distance_moduli_fn(z,h_proposed,omega_m_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain1[count-1]=h_current
    omega_m_values_chain1[count-1]=omega_m_current
corner_data_omega_m[0:500]=omega_m_values_chain1[length-500:length]
corner_data_h[0:500]=h_values_chain1[length-500:length]

print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
adap_cov_h_start1 = h_current
adap_cov_omega_m_start1 = omega_m_current


#Chain-2
count =1;
h_current = np.random.uniform()
omega_m_current = np.random.uniform()
distance_moduli_theo_c = distance_moduli_fn(z,h_current,omega_m_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current)
h_values_chain2 = np.zeros(length)
omega_m_values_chain2 = np.zeros(length)
h_values_chain2[count-1] = h_current
omega_m_values_chain2[count-1] = omega_m_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1):
        distance_moduli_theo_p = distance_moduli_fn(z,h_proposed,omega_m_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain2[count-1]=h_current
    omega_m_values_chain2[count-1]=omega_m_current
corner_data_omega_m[500:1000]=omega_m_values_chain2[length-500:length]
corner_data_h[500:1000]=h_values_chain2[length-500:length]

print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
adap_cov_h_start2 = h_current
adap_cov_omega_m_start2 = omega_m_current


#Chain-3
count =1;
h_current = np.random.uniform()
omega_m_current = np.random.uniform()
distance_moduli_theo_c = distance_moduli_fn(z,h_current,omega_m_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current)
h_values_chain3 = np.zeros(length)
omega_m_values_chain3 = np.zeros(length)
h_values_chain3[count-1] = h_current
omega_m_values_chain3[count-1] = omega_m_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1):
        distance_moduli_theo_p = distance_moduli_fn(z,h_proposed,omega_m_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain3[count-1]=h_current
    omega_m_values_chain3[count-1]=omega_m_current
corner_data_omega_m[1000:1500]=omega_m_values_chain3[length-500:length]
corner_data_h[1000:1500]=h_values_chain3[length-500:length]
    
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
adap_cov_h_start3 = h_current
adap_cov_omega_m_start3 = omega_m_current


#Chain-4

count =1;
h_current = np.random.uniform()
omega_m_current = np.random.uniform()
distance_moduli_theo_c = distance_moduli_fn(z,h_current,omega_m_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current)
h_values_chain4 = np.zeros(length)
omega_m_values_chain4 = np.zeros(length)
h_values_chain4[count-1] = h_current
omega_m_values_chain4[count-1] = omega_m_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1):
        distance_moduli_theo_p = distance_moduli_fn(z,h_proposed,omega_m_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain4[count-1]=h_current
    omega_m_values_chain4[count-1]=omega_m_current
corner_data_omega_m[1500:2000]=omega_m_values_chain4[length-500:length]
corner_data_h[1500:2000]=h_values_chain4[length-500:length]
    
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)

adap_cov_h_start4 = h_current
adap_cov_omega_m_start4 = omega_m_current

h_mean = np.mean(corner_data_h)
omega_m_mean = np.mean(corner_data_omega_m)
parameters_mean = np.array([h_mean,omega_m_mean])
covariance_mat = np.cov(np.stack((corner_data_h,corner_data_omega_m)))
#The following Covariance differs from that read from the file
print("Covariance Matrix=")
print(covariance_mat)


length_adap_cov = 2000
h_values_final = np.zeros(length_adap_cov*4)
omega_m_values_final = np.zeros(length_adap_cov*4)

#Adaptive Covariance - First Chain
count=1;
adap_cov_h_values1 = np.zeros(length_adap_cov)
adap_cov_omega_m_values1 = np.zeros(length_adap_cov)
distance_moduli_mean = distance_moduli_fn(z,adap_cov_h_start1,adap_cov_omega_m_start1)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start1,adap_cov_omega_m_start1]),covariance_mat)
adap_cov_h_values1[count-1] = parameters_current[0]
adap_cov_omega_m_values1[count-1] = parameters_current[1]
distance_moduli_model = distance_moduli_fn(z,parameters_current[0],parameters_current[1])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    distance_moduli_model = distance_moduli_fn(z,parameters_proposed[0],parameters_proposed[1])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1])
    if(p_proposed > p_current):
        parameters_current = parameters_proposed
        p_current = p_proposed
    elif(np.exp(p_current)!=0):
        p_transition = p_proposed-p_current
        p_random = np.random.uniform()
        if(p_random <= np.exp(p_transition)):
            parameters_current = parameters_proposed
            p_current = p_proposed
    count = count+1;
    adap_cov_h_values1[count-1] = parameters_current[0]
    adap_cov_omega_m_values1[count-1] = parameters_current[1]

        
print("Adaptive Covariance Chain-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])

#Adaptive Covariance - Second Chain
count=1;
adap_cov_h_values2 = np.zeros(length_adap_cov)
adap_cov_omega_m_values2 = np.zeros(length_adap_cov)
distance_moduli_mean = distance_moduli_fn(z,adap_cov_h_start2,adap_cov_omega_m_start2)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start2,adap_cov_omega_m_start2]),covariance_mat)
adap_cov_h_values2[count-1] = parameters_current[0]
adap_cov_omega_m_values2[count-1] = parameters_current[1]
distance_moduli_model = distance_moduli_fn(z,parameters_current[0],parameters_current[1])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    distance_moduli_model = distance_moduli_fn(z,parameters_proposed[0],parameters_proposed[1])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1])
    if(p_proposed > p_current):
        parameters_current = parameters_proposed
        p_current = p_proposed
    elif(np.exp(p_current)!=0):
        p_transition = p_proposed-p_current
        p_random = np.random.uniform()
        if(p_random <= np.exp(p_transition)):
            parameters_current = parameters_proposed
            p_current = p_proposed
    count = count+1;
    adap_cov_h_values2[count-1] = parameters_current[0]
    adap_cov_omega_m_values2[count-1] = parameters_current[1]

        
print("Adaptive Covariance Chain-2-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])

#Adaptive Covariance - Third Chain
count=1;
adap_cov_h_values3 = np.zeros(length_adap_cov)
adap_cov_omega_m_values3 = np.zeros(length_adap_cov)
distance_moduli_mean = distance_moduli_fn(z,adap_cov_h_start3,adap_cov_omega_m_start3)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start3,adap_cov_omega_m_start3]),covariance_mat)
adap_cov_h_values1[count-1] = parameters_current[0]
adap_cov_omega_m_values1[count-1] = parameters_current[1]
distance_moduli_model = distance_moduli_fn(z,parameters_current[0],parameters_current[1])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    distance_moduli_model = distance_moduli_fn(z,parameters_proposed[0],parameters_proposed[1])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1])
    if(p_proposed > p_current):
        parameters_current = parameters_proposed
        p_current = p_proposed
    elif(np.exp(p_current)!=0):
        p_transition = p_proposed-p_current
        p_random = np.random.uniform()
        if(p_random <= np.exp(p_transition)):
            parameters_current = parameters_proposed
            p_current = p_proposed
    count = count+1;
    adap_cov_h_values3[count-1] = parameters_current[0]
    adap_cov_omega_m_values3[count-1] = parameters_current[1]

        
print("Adaptive Covariance Chain-3-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])

#Adaptive Covariance - Fourth Chain
count=1;
adap_cov_h_values4 = np.zeros(length_adap_cov)
adap_cov_omega_m_values4 = np.zeros(length_adap_cov)
distance_moduli_mean = distance_moduli_fn(z,adap_cov_h_start4,adap_cov_omega_m_start4)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start4,adap_cov_omega_m_start4]),covariance_mat)
adap_cov_h_values4[count-1] = parameters_current[0]
adap_cov_omega_m_values4[count-1] = parameters_current[1]
distance_moduli_model = distance_moduli_fn(z,parameters_current[0],parameters_current[1])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    distance_moduli_model = distance_moduli_fn(z,parameters_proposed[0],parameters_proposed[1])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1])
    if(p_proposed > p_current):
        parameters_current = parameters_proposed
        p_current = p_proposed
    elif(np.exp(p_current)!=0):
        p_transition = p_proposed-p_current
        p_random = np.random.uniform()
        if(p_random <= np.exp(p_transition)):
            parameters_current = parameters_proposed
            p_current = p_proposed
    count = count+1;
    adap_cov_h_values4[count-1] = parameters_current[0]
    adap_cov_omega_m_values4[count-1] = parameters_current[1]

        
print("Adaptive Covariance Chain-4-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])

h_values_final = np.hstack((adap_cov_h_values1,adap_cov_h_values2,adap_cov_h_values3,adap_cov_h_values4))
omega_m_values_final = np.hstack((adap_cov_omega_m_values1,adap_cov_omega_m_values2,adap_cov_omega_m_values3,adap_cov_omega_m_values4))
p_max = likelihood(distance_moduli_fn(z,h_values_final[0],omega_m_values_final[0])) + prior(h_values_final[0],omega_m_values_final[0])
h_max = h_values_final[0]
omega_m_max = omega_m_values_final[0]
for i in range(1,length_adap_cov*4):
    p_current = likelihood(distance_moduli_fn(z,h_values_final[i],omega_m_values_final[i]))
    if(p_current > p_max):
        p_max = p_current
        h_max = h_values_final[i]
        omega_m_max = omega_m_values_final[i]

print("---------Final Values----------")
print("h=")
print(h_max)
print("omega_m")
print(omega_m_max)
        
#We now find the most appropriate point

plt.figure(1)
plt.plot(np.arange(length),h_values_chain1);
plt.plot(np.arange(length),h_values_chain2);
plt.plot(np.arange(length),h_values_chain3);
plt.plot(np.arange(length),h_values_chain4);
plt.figure(2)
plt.plot(np.arange(length),omega_m_values_chain1);
plt.plot(np.arange(length),omega_m_values_chain2);
plt.plot(np.arange(length),omega_m_values_chain3);
plt.plot(np.arange(length),omega_m_values_chain4);
plt.figure(3)
plt.plot(np.arange(length_adap_cov),adap_cov_h_values1);
plt.plot(np.arange(length_adap_cov),adap_cov_h_values2);
plt.plot(np.arange(length_adap_cov),adap_cov_h_values3);
plt.plot(np.arange(length_adap_cov),adap_cov_h_values4);
plt.figure(4)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values1);
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values2);
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values3);
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values4);
data1 = np.column_stack((corner_data_omega_m,corner_data_h))
data2 = np.column_stack((omega_m_values_final,h_values_final))
levels = 1.0 - np.exp(-0.5*np.arange(1.0,2.1,0.5)**2)
fig=plt.figure(13)
ax = fig.add_subplot(111)
ax.set_ylabel("Distance Modulus")
ax.set_xlabel("Redshift")
plt.plot(z,distance_moduli_data,'o')
plt.errorbar(z,distance_moduli_data,yerr=np.sqrt(np.diag(cov_matrix)),linestyle="none")
distance_moduli_model = distance_moduli_fn(z,h_max,omega_m_max)
plt.plot(z,distance_moduli_model)
figure1 = corner.corner(data1, bins = 25, levels = levels, smooth = 1.0, smooth1d = 1.0,labels = ["omega_m","h"])
figure2 = corner.corner(data2, bins = 30, levels = levels, smooth = 1.0, smooth1d = 1.0, show_titles = True,\
                        label_kwargs = dict(size = '20'),title_kwargs = dict(size='20'),fill_contours = True,\
                        quantiles = [0.158656,0.841344],labels = [r"$\Omega_m$","h"],range = [[0.1,0.5],[0.64,0.76]],truths = [omega_m_max,h_max])


plt.show()
