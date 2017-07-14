# MCMC Technique to fit a curve through available supernovae data
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import scipy.special as sc
import corner

#The data points are stored in the 'jla_mub_0.txt' file
z_data=np.zeros(31); distance_moduli_data=np.zeros(31)
file = open("jla_mub_0.txt",'r')
count =0
for line in file:
    z_data[count], distance_moduli_data[count] = line.split(' ', 1)
    count = count +1
corner_data_omega_m = np.zeros(2000);
corner_data_h = np.zeros(2000);
corner_data_omega_v = np.zeros(2000);
corner_data_omega_m_adap_cov = np.zeros(2000);
corner_data_h_adap_cov = np.zeros(2000);
corner_data_omega_v_adap_cov = np.zeros(2000);


#We have 31 data points, and hence we have a corresponding 31 X 31 Covariance matrix
cov_matrix=np.zeros(31*31); 
with open('jla_mub_covmatrix.txt') as f:
  cov_matrix = [float(i) for i in f]
cov_matrix=np.reshape(cov_matrix,(31,31))
sigma_h = 0.025
sigma_omega_m = 0.025
sigma_omega_v = 0.025
#We need to calculate the theoretical values as well, hence we need a function for that
distance_moduli_theo_c = np.zeros(31)
distance_moduli_theo_p = np.zeros(31)

def data_function(z,h,omega_m,omega_v):
    return 1/(100*h*np.sqrt(omega_m*(1+z)*(1+z)*(1+z) + omega_v))

def distance_moduli_fn(z,h,omega_m,omega_v):
    a = 1.0/(1+z)
    k = -1*(1-omega_m-omega_v)*a*a*100*100*h*h/(300000*300000)
    length_int = 500
    z_int = np.linspace(0,z,num=length_int)
    chi_points = data_function(z_int,h,omega_m,omega_v)
    comoving_distance = 300000*np.trapz(chi_points,x=z_int)
    if(k>0):
        metric_distance = np.sin(np.sqrt(k)*comoving_distance)/np.sqrt(k)
    elif(k<0):
        metric_distance = np.sinh(np.sqrt(np.abs(k))*comoving_distance)/np.sqrt(np.abs(k))
    else:
        metric_distance = comoving_distance
    distance_modulus = 25 + 5*np.log10(np.abs(metric_distance)*(1+z))
    return distance_modulus;

def likelihood(distance_moduli):
    p=multivariate_normal.pdf(distance_moduli,distance_moduli_data,cov_matrix)
    if(p!=0):
        return np.log(p);
    else:
        return (-np.inf);

def prior(h,omega_m,omega_v):
    if(h>=0.6 and h<=0.8 and omega_m>=0.2 and omega_m<=0.4 and omega_v>=0.6 and omega_v<=0.8):
        return 0;
    else:
        return (-np.inf);

def likelihood_adap_cov(distance_moduli,distance_moduli_mean):
    p=multivariate_normal.pdf(distance_moduli,distance_moduli_mean,cov_matrix)
    if(p!=0):
        return np.log(p);
    else:
        return (-np.inf);

#Chain - 1
length = 3000
count =1;
h_current = np.random.uniform(0.6,0.8)
omega_m_current = np.random.uniform(0.2,0.4)
omega_v_current = np.random.uniform(0.6,0.8)
for i in range(0,31):
    distance_moduli_theo_c[i] = distance_moduli_fn(z_data[i],h_current,omega_m_current,omega_v_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current,omega_v_current)
h_values_chain1 = np.zeros(length)
omega_m_values_chain1 = np.zeros(length)
omega_v_values_chain1 = np.zeros(length)
h_values_chain1[count-1] = h_current
omega_m_values_chain1[count-1] = omega_m_current
omega_v_values_chain1[count-1] = omega_v_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    omega_v_proposed = np.random.normal(omega_v_current,sigma_omega_v)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1 and omega_v_proposed >=0 and omega_v_proposed <=1):
        for i in range(0,31):
            distance_moduli_theo_p[i] = distance_moduli_fn(z_data[i],h_proposed,omega_m_proposed,omega_v_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed,omega_v_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            omega_v_current = omega_v_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                omega_v_current = omega_v_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain1[count-1]=h_current
    omega_m_values_chain1[count-1]=omega_m_current
    omega_v_values_chain1[count-1]=omega_v_current
corner_data_omega_m[0:500]=omega_m_values_chain1[length-500:length]
corner_data_omega_v[0:500]=omega_v_values_chain1[length-500:length]
corner_data_h[0:500]=h_values_chain1[length-500:length]

print("MCMC Chain-1-")
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
print("omega_v =")
print(omega_v_current)
print("omega_k =")
print(1-omega_m_current-omega_v_current)

adap_cov_h_start1 = h_current
adap_cov_omega_m_start1 = omega_m_current
adap_cov_omega_v_start1 = omega_v_current


#Chain - 2

count =1;
h_current = np.random.uniform(0.6,0.8)
omega_m_current = np.random.uniform(0.2,0.4)
omega_v_current = np.random.uniform(0.6,0.8)
for i in range(0,31):
    distance_moduli_theo_c[i] = distance_moduli_fn(z_data[i],h_current,omega_m_current,omega_v_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current,omega_v_current)
h_values_chain2 = np.zeros(length)
omega_m_values_chain2 = np.zeros(length)
omega_v_values_chain2 = np.zeros(length)
h_values_chain2[count-1] = h_current
omega_m_values_chain2[count-1] = omega_m_current
omega_v_values_chain2[count-1] = omega_v_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    omega_v_proposed = np.random.normal(omega_v_current,sigma_omega_v)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1 and omega_v_proposed >=0 and omega_v_proposed <=1):
        for i in range(0,31):
            distance_moduli_theo_p[i] = distance_moduli_fn(z_data[i],h_proposed,omega_m_proposed,omega_v_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed,omega_v_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            omega_v_current = omega_v_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                omega_v_current = omega_v_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain2[count-1]=h_current
    omega_m_values_chain2[count-1]=omega_m_current
    omega_v_values_chain2[count-1]=omega_v_current
corner_data_omega_m[500:1000]=omega_m_values_chain2[length-500:length]
corner_data_omega_v[500:1000]=omega_v_values_chain2[length-500:length]
corner_data_h[500:1000]=h_values_chain2[length-500:length]

print("MCMC Chain-2-")
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
print("omega_v =")
print(omega_v_current)
print("omega_k =")
print(1-omega_m_current-omega_v_current)

adap_cov_h_start2 = h_current
adap_cov_omega_m_start2 = omega_m_current
adap_cov_omega_v_start2 = omega_v_current

#Chain - 3

count =1;
h_current = np.random.uniform(0.6,0.8)
omega_m_current = np.random.uniform(0.2,0.4)
omega_v_current = np.random.uniform(0.6,0.8)
for i in range(0,31):
    distance_moduli_theo_c[i] = distance_moduli_fn(z_data[i],h_current,omega_m_current,omega_v_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current,omega_v_current)
h_values_chain3 = np.zeros(length)
omega_m_values_chain3 = np.zeros(length)
omega_v_values_chain3 = np.zeros(length)
h_values_chain3[count-1] = h_current
omega_m_values_chain3[count-1] = omega_m_current
omega_v_values_chain3[count-1] = omega_v_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    omega_v_proposed = np.random.normal(omega_v_current,sigma_omega_v)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1 and omega_v_proposed >=0 and omega_v_proposed <=1):
        for i in range(0,31):
            distance_moduli_theo_p[i] = distance_moduli_fn(z_data[i],h_proposed,omega_m_proposed,omega_v_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed,omega_v_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            omega_v_current = omega_v_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                omega_v_current = omega_v_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain3[count-1]=h_current
    omega_m_values_chain3[count-1]=omega_m_current
    omega_v_values_chain3[count-1]=omega_v_current
corner_data_omega_m[1000:1500]=omega_m_values_chain3[length-500:length]
corner_data_omega_v[1000:1500]=omega_v_values_chain3[length-500:length]
corner_data_h[1000:1500]=h_values_chain3[length-500:length]

print("MCMC Chain-3-")
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
print("omega_v =")
print(omega_v_current)
print("omega_k =")
print(1-omega_m_current-omega_v_current)

adap_cov_h_start3 = h_current
adap_cov_omega_m_start3 = omega_m_current
adap_cov_omega_v_start3 = omega_v_current

#Chain - 4

count =1;
h_current = np.random.uniform(0.6,0.8)
omega_m_current = np.random.uniform(0.2,0.4)
omega_v_current = np.random.uniform(0.6,0.8)
for i in range(0,31):
    distance_moduli_theo_c[i] = distance_moduli_fn(z_data[i],h_current,omega_m_current,omega_v_current)
p_current = likelihood(distance_moduli_theo_c) + prior(h_current,omega_m_current,omega_v_current)
h_values_chain4 = np.zeros(length)
omega_m_values_chain4 = np.zeros(length)
omega_v_values_chain4 = np.zeros(length)
h_values_chain4[count-1] = h_current
omega_m_values_chain4[count-1] = omega_m_current
omega_v_values_chain4[count-1] = omega_v_current

while (count<length):
    h_proposed = np.random.normal(h_current,sigma_h)
    omega_m_proposed = np.random.normal(omega_m_current,sigma_omega_m)
    omega_v_proposed = np.random.normal(omega_v_current,sigma_omega_v)
    if(omega_m_proposed >=0 and omega_m_proposed <=1 and h_proposed >=0 and h_proposed <=1 and omega_v_proposed >=0 and omega_v_proposed <=1):
        for i in range(0,31):
            distance_moduli_theo_p[i] = distance_moduli_fn(z_data[i],h_proposed,omega_m_proposed,omega_v_proposed)
        p_proposed = likelihood(distance_moduli_theo_p) + prior(h_proposed,omega_m_proposed,omega_v_proposed);
        if(p_proposed >= p_current):
            h_current = h_proposed
            omega_m_current = omega_m_proposed
            omega_v_current = omega_v_proposed
            p_current = p_proposed
        elif(np.exp(p_current) != 0):
            p_transition = p_proposed - p_current
            p_random = np.random.uniform();
            if(p_random <= np.exp(p_transition)): 
                h_current = h_proposed
                omega_m_current = omega_m_proposed
                omega_v_current = omega_v_proposed
                p_current = p_proposed
        count = count+1;
    h_values_chain4[count-1]=h_current
    omega_m_values_chain4[count-1]=omega_m_current
    omega_v_values_chain4[count-1]=omega_v_current
corner_data_omega_m[1500:2000]=omega_m_values_chain4[length-500:length]
corner_data_omega_v[1500:2000]=omega_v_values_chain4[length-500:length]
corner_data_h[1500:2000]=h_values_chain4[length-500:length]

print("MCMC Chain-4-")
print("h=")
print(h_current)
print("omega_m =")
print(omega_m_current)
print("omega_v =")
print(omega_v_current)
print("omega_k =")
print(1-omega_m_current-omega_v_current)

adap_cov_h_start4 = h_current
adap_cov_omega_m_start4 = omega_m_current
adap_cov_omega_v_start4 = omega_v_current

covariance_mat = np.cov(np.stack((corner_data_h,corner_data_omega_m,corner_data_omega_v)))
#The follwoing Covariance differs from that read from the file
print("Covariance Matrix=")
print(covariance_mat)

length_adap_cov = 2000
#Adaptive Covariance - First Chain
count=1;
adap_cov_h_values1 = np.zeros(length_adap_cov)
adap_cov_omega_m_values1 = np.zeros(length_adap_cov)
adap_cov_omega_v_values1 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov_h_start1,adap_cov_omega_m_start1,adap_cov_omega_v_start1)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start1,adap_cov_omega_m_start1,adap_cov_omega_v_start1]),covariance_mat)
adap_cov_h_values1[count-1] = parameters_current[0]
adap_cov_omega_m_values1[count-1] = parameters_current[1]
adap_cov_omega_v_values1[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov_omega_v_values1[count-1] = parameters_current[2]

corner_data_omega_m_adap_cov[0:500]=adap_cov_omega_m_values1[length_adap_cov-500:length_adap_cov]
corner_data_omega_v_adap_cov[0:500]=adap_cov_omega_v_values1[length_adap_cov-500:length_adap_cov]
corner_data_h_adap_cov[0:500]=adap_cov_h_values1[length_adap_cov-500:length_adap_cov]
print("Adaptive Covariance Chain-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])
adap_cov2_h_start1 = parameters_current[0]
adap_cov2_omega_m_start1 = parameters_current[1]
adap_cov2_omega_v_start1 = parameters_current[2]

#Adaptive Covariance - Second Chain
count=1;
adap_cov_h_values2 = np.zeros(length_adap_cov)
adap_cov_omega_m_values2 = np.zeros(length_adap_cov)
adap_cov_omega_v_values2= np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov_h_start2,adap_cov_omega_m_start2,adap_cov_omega_v_start2)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start2,adap_cov_omega_m_start2,adap_cov_omega_v_start2]),covariance_mat)
adap_cov_h_values2[count-1] = parameters_current[0]
adap_cov_omega_m_values2[count-1] = parameters_current[1]
adap_cov_omega_v_values2[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov_omega_v_values2[count-1] = parameters_current[2]


corner_data_omega_m_adap_cov[500:1000]=adap_cov_omega_m_values2[length_adap_cov-500:length_adap_cov]
corner_data_omega_v_adap_cov[500:1000]=adap_cov_omega_v_values2[length_adap_cov-500:length_adap_cov]
corner_data_h_adap_cov[500:1000]=adap_cov_h_values2[length_adap_cov-500:length_adap_cov]
        
print("Adaptive Covariance Chain-2-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])
adap_cov2_h_start2 = parameters_current[0]
adap_cov2_omega_m_start2 = parameters_current[1]
adap_cov2_omega_v_start2 = parameters_current[2]

#Adaptive Covariance - Third Chain
count=1;
adap_cov_h_values3 = np.zeros(length_adap_cov)
adap_cov_omega_m_values3 = np.zeros(length_adap_cov)
adap_cov_omega_v_values3 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov_h_start3,adap_cov_omega_m_start3,adap_cov_omega_v_start3)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start3,adap_cov_omega_m_start3,adap_cov_omega_v_start3]),covariance_mat)
adap_cov_h_values3[count-1] = parameters_current[0]
adap_cov_omega_m_values3[count-1] = parameters_current[1]
adap_cov_omega_v_values3[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov_omega_v_values3[count-1] = parameters_current[2]


corner_data_omega_m_adap_cov[1000:1500]=adap_cov_omega_m_values3[length_adap_cov-500:length_adap_cov]
corner_data_omega_v_adap_cov[1000:1500]=adap_cov_omega_v_values3[length_adap_cov-500:length_adap_cov]
corner_data_h_adap_cov[1000:1500]=adap_cov_h_values3[length_adap_cov-500:length_adap_cov]

print("Adaptive Covariance Chain-3-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])
adap_cov2_h_start3 = parameters_current[0]
adap_cov2_omega_m_start3 = parameters_current[1]
adap_cov2_omega_v_start3 = parameters_current[2]

#Adaptive Covariance - Fourth Chain
count=1;
adap_cov_h_values4 = np.zeros(length_adap_cov)
adap_cov_omega_m_values4 = np.zeros(length_adap_cov)
adap_cov_omega_v_values4 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov_h_start4,adap_cov_omega_m_start4,adap_cov_omega_v_start4)
parameters_current = np.random.multivariate_normal(np.array([adap_cov_h_start4,adap_cov_omega_m_start4,adap_cov_omega_v_start4]),covariance_mat)
adap_cov_h_values1[count-1] = parameters_current[0]
adap_cov_omega_m_values4[count-1] = parameters_current[1]
adap_cov_omega_v_values4[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov_omega_v_values4[count-1] = parameters_current[2]

corner_data_omega_m_adap_cov[1500:2000]=adap_cov_omega_m_values4[length_adap_cov-500:length_adap_cov]
corner_data_omega_v_adap_cov[1500:2000]=adap_cov_omega_v_values4[length_adap_cov-500:length_adap_cov]
corner_data_h_adap_cov[1500:2000]=adap_cov_h_values4[length_adap_cov-500:length_adap_cov]
        
print("Adaptive Covariance Chain-4-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])
adap_cov2_h_start4 = parameters_current[0]
adap_cov2_omega_m_start4 = parameters_current[1]
adap_cov2_omega_v_start4 = parameters_current[2]


covariance_mat_adap_cov = np.cov(np.stack((corner_data_h_adap_cov,corner_data_omega_m_adap_cov,corner_data_omega_v_adap_cov)))
#The follwoing Covariance differs from that read from the file
print("Covariance Matrix=")
print(covariance_mat_adap_cov)

length_adap_cov2 = 2000
#Adaptive Covariance Again - First Chain
count=1;
adap_cov2_h_values1 = np.zeros(length_adap_cov)
adap_cov2_omega_m_values1 = np.zeros(length_adap_cov)
adap_cov2_omega_v_values1 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov2_h_start1,adap_cov2_omega_m_start1,adap_cov2_omega_v_start1)
parameters_current = np.random.multivariate_normal(np.array([adap_cov2_h_start1,adap_cov2_omega_m_start1,adap_cov2_omega_v_start1]),covariance_mat_adap_cov)
adap_cov2_h_values1[count-1] = parameters_current[0]
adap_cov2_omega_m_values1[count-1] = parameters_current[1]
adap_cov2_omega_v_values1[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov2_h_values1[count-1] = parameters_current[0]
    adap_cov2_omega_m_values1[count-1] = parameters_current[1]
    adap_cov2_omega_v_values1[count-1] = parameters_current[2]

print("Adaptive Covariance Chain(Again)-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])

#Adaptive Covariance Again - Second Chain
count=1;
adap_cov2_h_values2 = np.zeros(length_adap_cov)
adap_cov2_omega_m_values2 = np.zeros(length_adap_cov)
adap_cov2_omega_v_values2 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov2_h_start2,adap_cov2_omega_m_start2,adap_cov2_omega_v_start2)
parameters_current = np.random.multivariate_normal(np.array([adap_cov2_h_start2,adap_cov2_omega_m_start2,adap_cov2_omega_v_start2]),covariance_mat_adap_cov)
adap_cov2_h_values2[count-1] = parameters_current[0]
adap_cov2_omega_m_values2[count-1] = parameters_current[1]
adap_cov2_omega_v_values2[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov2_h_values2[count-1] = parameters_current[0]
    adap_cov2_omega_m_values2[count-1] = parameters_current[1]
    adap_cov2_omega_v_values2[count-1] = parameters_current[2]

print("Adaptive Covariance Chain(Again)-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])


#Adaptive Covariance Again - Third Chain
count=1;
adap_cov2_h_values3 = np.zeros(length_adap_cov)
adap_cov2_omega_m_values3 = np.zeros(length_adap_cov)
adap_cov2_omega_v_values3 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov2_h_start3,adap_cov2_omega_m_start3,adap_cov2_omega_v_start3)
parameters_current = np.random.multivariate_normal(np.array([adap_cov2_h_start3,adap_cov2_omega_m_start3,adap_cov2_omega_v_start3]),covariance_mat_adap_cov)
adap_cov2_h_values3[count-1] = parameters_current[0]
adap_cov2_omega_m_values3[count-1] = parameters_current[1]
adap_cov2_omega_v_values3[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov2_h_values3[count-1] = parameters_current[0]
    adap_cov2_omega_m_values3[count-1] = parameters_current[1]
    adap_cov2_omega_v_values3[count-1] = parameters_current[2]

print("Adaptive Covariance Chain(Again)-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])


#Adaptive Covariance Again - Fourth Chain
count=1;
adap_cov2_h_values4 = np.zeros(length_adap_cov)
adap_cov2_omega_m_values4 = np.zeros(length_adap_cov)
adap_cov2_omega_v_values4 = np.zeros(length_adap_cov)
distance_moduli_mean = np.zeros(31)
for i in range(0,31):
    distance_moduli_mean[i] = distance_moduli_fn(z_data[i],adap_cov2_h_start4,adap_cov2_omega_m_start4,adap_cov2_omega_v_start4)
parameters_current = np.random.multivariate_normal(np.array([adap_cov2_h_start4,adap_cov2_omega_m_start4,adap_cov2_omega_v_start4]),covariance_mat_adap_cov)
adap_cov2_h_values4[count-1] = parameters_current[0]
adap_cov2_omega_m_values4[count-1] = parameters_current[1]
adap_cov2_omega_v_values4[count-1] = parameters_current[2]
distance_moduli_model = np.zeros(31)
for i in range(0,31):
    distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_current[0],parameters_current[1],parameters_current[2])
p_current = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_current[0],parameters_current[1],parameters_current[2])
while(count<length_adap_cov):
    parameters_proposed = np.random.multivariate_normal(parameters_current,covariance_mat)
    for i in range(0,31):
        distance_moduli_model[i] = distance_moduli_fn(z_data[i],parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
    p_proposed = likelihood_adap_cov(distance_moduli_model,distance_moduli_mean) + prior(parameters_proposed[0],parameters_proposed[1],parameters_proposed[2])
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
    adap_cov2_h_values4[count-1] = parameters_current[0]
    adap_cov2_omega_m_values4[count-1] = parameters_current[1]
    adap_cov2_omega_v_values4[count-1] = parameters_current[2]

print("Adaptive Covariance Chain(Again)-1-")
print("h=")
print(parameters_current[0])
print("omega_m =")
print(parameters_current[1])
print("omega_v =")
print(parameters_current[2])
print("omega_k =")
print(1-parameters_current[1]-parameters_current[2])


h_values_final = np.zeros(length_adap_cov2)
omega_m_values_final = np.zeros(length_adap_cov2)
omega_v_values_final = np.zeros(length_adap_cov2)
h_values_final = np.hstack((adap_cov2_h_values1[length_adap_cov2-500:length_adap_cov2],adap_cov2_h_values2[length_adap_cov2-500:length_adap_cov2],adap_cov2_h_values3[length_adap_cov2-500:length_adap_cov2],adap_cov2_h_values4[length_adap_cov2-500:length_adap_cov2]))
omega_m_values_final = np.hstack((adap_cov_omega_m_values1[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_m_values2[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_m_values3[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_m_values4[length_adap_cov2-500:length_adap_cov2]))
omega_v_values_final = np.hstack((adap_cov_omega_v_values1[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_v_values2[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_v_values3[length_adap_cov2-500:length_adap_cov2],adap_cov_omega_v_values4[length_adap_cov2-500:length_adap_cov2]))
distance_moduli_calc = np.zeros(31)
for i in range(0,31):
    distance_moduli_calc[i] = distance_moduli_fn(z_data[i],h_values_final[0],omega_m_values_final[0],omega_v_values_final[0])
p_max = likelihood(distance_moduli_calc)
h_max = h_values_final[0]
omega_m_max = omega_m_values_final[0]
omega_v_max = omega_v_values_final[0]
for j in range(1,length_adap_cov):
    for i in range(0,31):
        distance_moduli_calc[i] = distance_moduli_fn(z_data[i],h_values_final[j],omega_m_values_final[j],omega_v_values_final[j])
    p_current = likelihood(distance_moduli_calc)
    if(p_current > p_max):
        p_max = p_current
        h_max = h_values_final[j]
        omega_m_max = omega_m_values_final[j]
        omega_v_max = omega_v_values_final[j]

print("h_values=")
print(h_values_final)
print("omega_m_values=")
print(omega_m_values_final)
print("omega_v_values=")
print(omega_v_values_final)

print("---------Final Values----------")
print("h=")
print(h_max)
print("omega_m")
print(omega_m_max)
print("omega_v")
print(omega_v_max)
print("omega_k=")
print(1-omega_m_max-omega_v_max)

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
plt.plot(np.arange(length),omega_v_values_chain1);
plt.plot(np.arange(length),omega_v_values_chain2);
plt.plot(np.arange(length),omega_v_values_chain3);
plt.plot(np.arange(length),omega_v_values_chain4);
plt.figure(4)
plt.plot(np.arange(length),1-omega_v_values_chain1-omega_m_values_chain1);
plt.plot(np.arange(length),1-omega_v_values_chain2-omega_m_values_chain2);
plt.plot(np.arange(length),1-omega_v_values_chain3-omega_m_values_chain3);
plt.plot(np.arange(length),1-omega_v_values_chain4-omega_m_values_chain4);
plt.figure(5)
plt.plot(np.arange(length_adap_cov),adap_cov_h_values1)
plt.plot(np.arange(length_adap_cov),adap_cov_h_values2)
plt.plot(np.arange(length_adap_cov),adap_cov_h_values3)
plt.plot(np.arange(length_adap_cov),adap_cov_h_values4)
plt.figure(6)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values1)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values2)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values3)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_m_values4)
plt.figure(7)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_v_values1)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_v_values2)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_v_values3)
plt.plot(np.arange(length_adap_cov),adap_cov_omega_v_values4)
plt.figure(8)
plt.plot(np.arange(length_adap_cov),1-adap_cov_omega_m_values1-adap_cov_omega_v_values1)
plt.plot(np.arange(length_adap_cov),1-adap_cov_omega_m_values2-adap_cov_omega_v_values2)
plt.plot(np.arange(length_adap_cov),1-adap_cov_omega_m_values3-adap_cov_omega_v_values3)
plt.plot(np.arange(length_adap_cov),1-adap_cov_omega_m_values4-adap_cov_omega_v_values4)
plt.figure(9)
plt.plot(np.arange(length_adap_cov2),adap_cov2_h_values1)
plt.plot(np.arange(length_adap_cov2),adap_cov2_h_values2)
plt.plot(np.arange(length_adap_cov2),adap_cov2_h_values3)
plt.plot(np.arange(length_adap_cov2),adap_cov2_h_values4)
plt.figure(10)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_m_values1)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_m_values2)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_m_values3)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_m_values4)
plt.figure(11)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_v_values1)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_v_values2)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_v_values3)
plt.plot(np.arange(length_adap_cov2),adap_cov2_omega_v_values4)
plt.figure(12)
plt.plot(np.arange(length_adap_cov2),1-adap_cov2_omega_m_values1-adap_cov2_omega_v_values1)
plt.plot(np.arange(length_adap_cov2),1-adap_cov2_omega_m_values2-adap_cov2_omega_v_values2)
plt.plot(np.arange(length_adap_cov2),1-adap_cov2_omega_m_values3-adap_cov2_omega_v_values3)
plt.plot(np.arange(length_adap_cov2),1-adap_cov2_omega_m_values4-adap_cov2_omega_v_values4)
data = np.column_stack((omega_m_values_final,omega_v_values_final,h_values_final))
levels = 1.0 - np.exp(-0.5*np.arange(1.0,2.1,0.5)**2)
fig=plt.figure(13)
ax = fig.add_subplot(111)
ax.set_ylabel("Distance Modulus")
ax.set_xlabel("Redshift")
plt.plot(z_data,distance_moduli_data,'o')
plt.errorbar(z_data,distance_moduli_data,yerr=np.sqrt(np.diag(cov_matrix)),linestyle="none")
for i in range(0,31):
        distance_moduli_calc[i] = distance_moduli_fn(z_data[i],h_max,omega_m_max,omega_v_max)
plt.plot(z_data,distance_moduli_calc)

figure1 = corner.corner(data, bins = 30,show_titles = True,\
                        label_kwargs = dict(size = '20'),title_kwargs = dict(size='20'),fill_contours = True, levels = levels, smooth = 1.0,\
                        quantiles = [0.158656,0.841344],labels = [r"$\Omega_m$",r"$\Omega_{\Lambda}$","h"], truths = [omega_m_max,omega_v_max,h_max],range = [[0.1,0.6],[0.35,0.95],[0.4,0.9]] ,smooth1d = 1.0)

plt.show()

