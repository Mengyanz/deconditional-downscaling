import numpy as np
import scipy.linalg as spla
from src.pes.utilities import *
import torch

#This files include functions to compute the covariances. The author's original implementations are available at 
#https://bitbucket.org/jmh233/codepesnips2014/src/master/sourceFiles/. Also, pleaser refer to (E. Solak, R. Murray-Smith, 
#W. E. Leithead, D. J. Leith, and C. E. Rasmussen. Derivative observations in Gaussian process models of dynamic systems. 
#In NIPS, pages 1057–1064, 2003.) for information of constructing kernel matrix including derivative observations.



#Calculate the covariance between x and y which are both in d dimension.
# def covariance(x, y, sigma, l):
#     distance = 0
#     for i in range(len(x)):
#         distance = distance + ((x[i] - y[i])**2)/((l[i])**2)
#     distance = -0.5*distance
        
#     cov = sigma * np.exp(distance)
#     return cov

def covariance(x, y, model, l):
    with torch.no_grad():
            # return model.individuals_kernel(torch.tensor(x),torch.tensor(y)).evaluate()
        return model.individuals_kernel(x,y).evaluate()
    
def bag_covariance(x, y, model):
    with torch.no_grad():
        return model.bag_kernel(x,y).evaluate()
    
def covariance_cross(nObservations, x_input, model,λ):
    x,y,tilde_y=nObservations
    # cov(f(x_input), z)
    with torch.no_grad():
        N = len(y)
        Lλ = model.bag_kernel(y, y).add_diag(λ * N * torch.ones(N))
        chol = torch.linalg.cholesky(Lλ.evaluate())
        Lys_ys2 = model.bag_kernel(y, tilde_y).evaluate()
        Lλinv_Lys_ys2 = torch.cholesky_solve(Lys_ys2, chol)
        Kxs = model.individuals_kernel(x_input, x)
        return (Kxs @ Lλinv_Lys_ys2).T
    
#Calculate the covariance between the derivative of x with respect to dimention m and y 
def cov_devX_y_cross(nObservations, x_input, model,λ,l,m):
    x,y,tilde_y=nObservations
    # result  = -((x_input[m] - xs[m])/((l[m])**2))*covariance_cross(ys2, x_input,xs,ys, model,λ)
    with torch.no_grad():
        N = len(y)
        Lλ = model.bag_kernel(y, y).add_diag(λ * N * torch.ones(N))
        chol = torch.linalg.cholesky(Lλ.evaluate())
        Lys_ys2 = model.bag_kernel(y, tilde_y).evaluate()
        Lλinv_Lys_ys2 = torch.cholesky_solve(Lys_ys2, chol)
        Kxs = model.individuals_kernel(x_input, x)
        # print('Kxs shape: ', Kxs.shape)
        dev = -((x_input[:,m] - x[:,m])/((l[:,m])**2)).reshape(1,-1)
        # print('dev shape: ', dev.shape)
        # print('dev * Kxs shape', (dev * Kxs).shape)
        return ((dev * Kxs) @ Lλinv_Lys_ys2)

#Calculate the covariance between x and the derivative of y with respect to dimention m  
def cov_x_devY_cross(nObservations, x_input, model,λ,l, m):
    result = -cov_devX_y_cross(nObservations, x_input, model,λ,l,m)
    return result

#Calculate the covariance between x and the second derivative of y with respect to m and n
def cov_devdevX_y_cross(nObservations, x_input, model,λ,l, m , n):
    xs,ys,ys2=nObservations
    result = 0
    if (m == n):
        result = -covariance_cross(nObservations, x_input, model, λ)/((l[m])**2) + ((x_input[m] - xs[m])/((l[m])**2))*cov_devX_y_cross(nObservations, x_input, model,λ,l,n)
    else:
        result = ((x_input[m] - xs[m])/((l[m])**2))*cov_devX_y_cross(nObservations, x_input, model,λ,l,n)
    return result

#---------------------------------------------------------------------------------------

#Calculate the covariance between the derivative of x with respect to dimention m and y 
def cov_devX_y(x, y, model, l, m):
    result  = -((x[m] - y[m])/((l[m])**2))*covariance(x,y,model,l)
    #result = 1
    return result


#Calculate the covariance between x and the derivative of y with respect to dimention m  
def cov_x_devY(x, y, model, l, m):
    result = -cov_devX_y(x,y, model, l, m)
    return result
    


#Calculate the covariance between the derivative of y with respect to n and the derivative of x with respect to m 
def cov_devY_devX(x,y, model, l , n, m):
    result = 0
    if (m == n):
        result = covariance(x, y, model, l)/((l[m])**2) + ((x[n] - y[n])/((l[n])**2))*cov_devX_y(x,y,model,l,m)
    else:
        result = ((x[n] - y[n])/((l[n])**2))*cov_devX_y(x,y,model,l,m)
    return result


#Calculate the covariance between x and the second derivative of y with respect to m and n
def cov_x_devdevY(x,y, sigma, l , m , n):
    result = 0
    if (m == n):
        result = -covariance(x, y, sigma, l)/((l[m])**2) + ((x[m] - y[m])/((l[m])**2))*cov_x_devY(x,y,sigma,l,n)
    else:
        result = ((x[m] - y[m])/((l[m])**2))*cov_x_devY(x,y,sigma,l,n)
    return result
    
    
    
#Calculate the covariance between the second derivative of x with respect to m and n and y
def cov_devdevX_y(x,y, model, l , m , n):
    result = 0
    if (m == n):
        result = -covariance(x, y, model, l)/((l[n])**2) - ((x[m] - y[m])/((l[m])**2))*cov_devX_y(x,y,model,l,n)
    else:
        result = -((x[m] - y[m])/((l[m])**2))*cov_devX_y(x,y,model,l,n)
    return result
    
    

    
#Calculate the covariance between the derivative of y with repect to k and the second derivative of x 
#with respect to m and n and 
def cov_devY_devdevX(x,y, model, l ,k, m , n):
    temp_result_1 = 0
    temp_result_2 = 0
    if (m == k):
        temp_result_1 = cov_devX_y(x,y,model,l,n)/((l[k])**2)
    if (n == k):
        temp_result_2 = cov_devX_y(x, y, model, l, m)/((l[k])**2)
        
    result = temp_result_1 + temp_result_2 + ((x[ k ] - y[ k ])/((l[k])**2)) * cov_devdevX_y(x, y, model, l,  m, n)
    return result

    
    
    
        
    
#Calculate the covariance between the second derivative of y with respect to m and n and the derivative of x
#with respect to k 
def cov_devdevY_devX(x,y, model, l , m , n, k):
    temp_result_1 = 0
    temp_result_2 = 0
    if (m == n):
        temp_result_1 = -cov_devX_y(x,y,model,l,k)/((l[m])**2)
    if (m == k):
        temp_result_2 = cov_x_devY(x, y, model, l, n)/((l[m])**2)
        
    result = temp_result_1 + temp_result_2 + ((x[m] - y[m])/((l[m])**2)) * cov_devY_devX(x, y, model, l,  n, k)    
    return result

    

    
    
    
    
#Calculate the covariance between the second derivative of y with respect to m and n and the second derivative of x
#with respect to i and j
def cov_devdevY_devdevX(x,y, model, l , m , n, i, j):
    temp_result_1 = 0
    temp_result_2 = 0
    temp_result_3 = 0
    if (m == n):
        temp_result_1 = -cov_devdevX_y(x,y,model,l,i, j)/((l[m])**2)
    if (i == m):
        temp_result_2 = cov_devY_devX(x, y, model, l, n, j)/((l[m])**2)
    if (j == m):
        temp_result_3 = cov_devY_devX(x, y, model, l, n, i)/((l[m])**2)
        
    result = temp_result_1 + temp_result_2 + temp_result_3 + ((x[m] - y[m])/((l[m])**2)) * cov_devY_devdevX(x, y, model, l,  n ,i, j)    
    
    return result
    

    

    
    
    
    
#############################
#############################
#Compute n observations and everything else



#Compute covariance matrix of n observations in d dimensionss
def covNobeservations(nObservations, num_of_obser, model, noise, l):
    xs,ys,ys2=nObservations
    cov_matrix = np.zeros((num_of_obser,num_of_obser))
    
    for i in range(num_of_obser):
        for j in range(i,num_of_obser):
            if i == j:
                cov_matrix[i,j] = bag_covariance(ys2[i], ys2[j], model) + noise
            else:
                cov_matrix[i,j] = bag_covariance(ys2[i], ys2[j], model)
                
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    return cov_matrix    
    


#Compute the part of the covariance matrix between n observations and the minimum
def cov_nObser_max(nObservations, value_at_max, num_of_obser, model, λ, l):
    xs,ys,ys2=nObservations
    cov_matrix = np.zeros((num_of_obser, 1))
    
    for i in range(num_of_obser):
        cov_matrix[i,0] = covariance_cross([xs, ys, ys2[i]], value_at_max, model, λ)
        
    return cov_matrix
    
    

    

#Compute the covariance matrix of n observations and gradient of the minimum
def cov_nObser_maxGrad(nObservations, value_at_max, num_of_obser, model, λ, l):
    xs,ys,ys2=nObservations
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_obser, d))
    for i in range(num_of_obser):
        for j in range(d):
            # TODO check here derivative is on x instead y
            # cov_matrix[i,j] = cov_x_devY_cross([xs, ys, ys2[i]], value_at_max, model,λ,l, j)
            cov_matrix[i,j] = cov_devX_y_cross([xs, ys, ys2[i]], value_at_max, model,λ,l, j)
    
    return cov_matrix



#Compute the covariance matrix of n observations and the hessian of the maxium(not sure works)
def cov_nObser_maxHess(nObservations, value_at_max, num_of_obser, model, noise, l):
    d = len(value_at_max)
    num_hessian_combo = int(d*(d+1)/2)
    cov_matrix = np.zeros((num_of_obser, num_hessian_combo))
    for i in range(num_of_obser):
        index = 0
        for j in range(d):
            for k in range(j,d):
                cov_matrix[i,index] = cov_x_devdevY( nObservations[i], value_at_max, model, l , j , k)
                index = index + 1
    
    return cov_matrix




#Compute the covariance matrix of n observations and the off-diagnal hessian of the maxium
def cov_nObser_off_maxHess(nObservations, value_at_max, num_of_obser, model,λ, l):
    xs,ys,ys2=nObservations
    d = len(value_at_max)
    num_hessian_combo = int(d*(d-1)/2)
    cov_matrix = np.zeros((num_of_obser, num_hessian_combo))
    for i in range(num_of_obser):
        index = 0
        for j in range(d):
            for k in range(j+1,d):
                cov_matrix[i,index] = cov_devdevX_y_cross([xs, ys, ys2[i]], value_at_max, model,λ,l, j , k)
                index = index + 1
    
    return cov_matrix                


#Compute the covariance matrix of n observations and the diagnal hessian of the maxium
def cov_nObser_diagonal_maxHess(nObservations, value_at_max, num_of_obser, model,λ, l):
    xs,ys,ys2=nObservations
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_obser, d))
    for i in range(num_of_obser):
        for j in range(d):
             # TODO: check devX_y instead of devY_x
            cov_matrix[i,j] = cov_devdevX_y_cross([xs, ys, ys2[i]], value_at_max, model,λ, l , j , j)
            
    return cov_matrix










###########################################
###########################################
#Compute gradient at minimum and everything else
    
    
    
#Compute the covariance matrix between gradient at minimum and gradient at minimum
def cov_maxGrad_maxGrad(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(i, d):
            cov_matrix[i,j] = cov_devY_devX(value_at_max, value_at_max, model, l , j , i)
    
    cov_matrix = cov_matrix + cov_matrix.T - np.diag(cov_matrix.diagonal())
    return cov_matrix




#Compute the covariance matrix between gradient at minimum and hessian at minimum
def cov_maxGrad_maxHess(value_at_max, model, l):
    d = len(value_at_max)
    num_hessian_combo = d*(d+1)/2
    cov_matrix = np.zeros((d,num_hessian_combo))
    for i in range(d):
        index = 0
        for j in range(d):
            for k in range(j,d):
                cov_matrix[i,index] = cov_devdevY_devX( value_at_max, value_at_max, model, l , j , k, i)
                index = index + 1
                
    return cov_matrix




#Compute the covariance matrix between gradient at minimum and off-diagonal hessian at minimum
def cov_maxGrad_off_maxHess(value_at_max, model, l):
    d = len(value_at_max)
    num_hessian_combo = int(d*(d-1)/2)
    cov_matrix = np.zeros((d,num_hessian_combo))
    for i in range(d):
        index = 0
        for j in range(d):
            for k in range(j+1,d):
                cov_matrix[i,index] = cov_devdevY_devX( value_at_max, value_at_max, model, l , j , k, i)
                index = index + 1
                           
    return cov_matrix    






#Compute the covariance matrix between gradient at minimum and diagonal hessian at minimum
def cov_maxGrad_diaHess(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            cov_matrix[i,j] = cov_devdevY_devX( value_at_max, value_at_max, model, l , j , j, i)
                
    return cov_matrix  






#Compute the covariance matrix between gradient at minimum and the minimum
def cov_maxGrad_max(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((d, 1))
        
    return cov_matrix









#########################################################
#########################################################
#Non-diagonal hessian and everything else




#Compute non-diagonal hessian and non-diagonal hessian
def cov_nonDiaHess_nonDiaHess(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((int(d*(d-1)/2),int(d*(d-1)/2)))
    index_1 = 0
    for m in range(d):
        for n in range(m+1,d):
            index_2 = 0
            for i in range(d):
                for j in range(i+1,d):
                    cov_matrix[index_1, index_2] = cov_devdevY_devdevX(value_at_max, value_at_max, model, l , i , j, m, n)
                    index_2 = index_2 + 1
            index_1 = index_1 + 1
            
    return cov_matrix








#Compute non-diagonal hessian and diagonal hessian
def cov_nonDiaHess_diaHess(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((int(d*(d-1)/2), d))
    index_1 = 0
    for m in range(d):
        for n in range(m+1,d):
            for i in range(d):
                cov_matrix[index_1, i] = cov_devdevY_devdevX(value_at_max, value_at_max, model, l, i, i, m, n)
            index_1 = index_1 + 1
            
    return cov_matrix





#Compute non-diagonal hessian and minimum
def cov_nonDiaHess_max(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((int(d*(d-1)/2)))
    index = 0
    for i in range(d):
        for j in range(i+1, d):
            cov_matrix[index] = cov_x_devdevY(value_at_max, value_at_max, model, l , i , j)
            index = index + 1
    
    return cov_matrix









##########################################################
##########################################################
#Diagonal hessian and everything else



#Compute the covariance matrix between diagonal hessian and diagonal hessian
def cov_diaHess_diaHess(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            cov_matrix[i,j] = cov_devdevY_devdevX(value_at_max, value_at_max, model, l, j, j, i, i)
    
    return cov_matrix






#Compute the covariance matrix between diagonal hessian and the minimum
def cov_diaHess_max(value_at_max, model, l):
    d = len(value_at_max)
    cov_matrix = np.zeros((d,1))
    for i in range(d):
        cov_matrix[i,0] = cov_x_devdevY(value_at_max, value_at_max, model, l , i , i)
        
    return cov_matrix
    

    
    
    
    
    



###################################################
###################################################

# minimum and everything else



#Compute the covariance between the minimum and the minimum
def cov_max_max(value_at_max, model, noise, l):
    cov_matrix = np.zeros((1,1))
    # TODO: check -- remove noise term
    # cov_matrix[0,0] = covariance(value_at_max, value_at_max, model, l) + noise
    cov_matrix[0,0] = covariance(value_at_max, value_at_max, model, l)
    return cov_matrix







##################################################
##################################################

#The points we want to evaluate xPrime and everything else



# #Compute the covariance between the x prime and n observations
# def cov_xPrime_nObservations(xPrime, nObservations, model, l):
#     num_of_xPrime = len(xPrime)
#     num_of_obser = len(nObservations)
#     cov_matrix = np.zeros((num_of_xPrime, num_of_obser))
#     for i in range(num_of_xPrime):
#         for j in range(num_of_obser):
#             cov_matrix[i,j] = covariance(xPrime[i], nObservations[j], model, l)
    
#     return cov_matrix

#Compute the covariance between the x prime and n observations
def cov_xPrime_nObservations(xPrime, nObservations, model, l):
    with torch.no_grad():
        xs, ys, ys2 = nObservations
        λ = 0.01
        N = len(ys)
        Lλ = model.bag_kernel(ys, ys).add_diag(λ * N * torch.ones(N))
        chol = torch.linalg.cholesky(Lλ.evaluate())
        Lys_ys2 = model.bag_kernel(ys, ys2).evaluate()
        Lλinv_Lys_ys2 = torch.cholesky_solve(Lys_ys2, chol)
        Kxs = model.individuals_kernel(xPrime, xs)
        return (Kxs @ Lλinv_Lys_ys2)



#Compute the covariance between the x prime and the gradient at the minimum
def cov_xPrime_maxGrad(xPrime, value_at_max, model, l):
    d = len(value_at_max)
    num_of_xPrime = len(xPrime)
    cov_matrix = np.zeros((num_of_xPrime, d))
    for i in range(num_of_xPrime):
        for j in range(d):
            cov_matrix[i,j] = cov_x_devY(xPrime[i], value_at_max, model, l, j)
            
    return cov_matrix




#Compute the covariance between the x prime and the off-diagonal hessian at the minimum
def cov_xPrime_nonDiaHess(xPrime, value_at_max, model, l):
    d = len(value_at_max)
    num_of_xPrime = len(xPrime)
    cov_matrix = np.zeros((num_of_xPrime, int(d*(d-1)/2)))
    for m in range(num_of_xPrime):
        index = 0
        for i in range(d):
            for j in range(i+1, d):
                cov_matrix[m, index] = cov_x_devdevY(xPrime[m], value_at_max, model, l , i , j)
                index = index + 1
                
    return cov_matrix




#Compute the covariance between the x prime and the diagonal hessian at the minimum
def cov_xPrime_diaHess(xPrime, value_at_max, model, l):
    d = len(value_at_max)
    num_of_xPrime = len(xPrime)
    cov_matrix = np.zeros((num_of_xPrime, d))
    for i in range(num_of_xPrime):
        for j in range(d):
            cov_matrix[i,j] = cov_x_devdevY(xPrime[i], value_at_max, model, l , j , j)
            
    return cov_matrix




#Compute the covariance between the x prime and the minimum
def cov_xPrime_max(xPrime, value_at_max, model, l):
    num_of_xPrime = len(xPrime)
    cov_matrix = np.zeros((num_of_xPrime, 1))
    for i in range(num_of_xPrime):
        cov_matrix[i,0] = covariance(xPrime[i], value_at_max, model, l)
        
    return cov_matrix
            





#Compute the covariance between the derivative of x prime with respect ot m and the n observations
def cov_gradXprime_nObser(xPrime, nObservations, model, l, m):
    num_of_xPrime = len(xPrime)
    num_of_obser = len(nObservations)
    cov_matrix = np.zeros((num_of_xPrime, num_of_obser))
    
    for i in range(num_of_xPrime):
        for j in range(num_of_obser):
            cov_matrix[i,j] = cov_devX_y(xPrime[i], nObservations[j], model, l, m)
            
    return cov_matrix




#Compute the covariance between the derivative of x prime with respect ot m and the gradient at the minimum
def cov_gradXprime_gradMax(xPrime, value_at_max, model, l, m):
    num_of_xPrime = len(xPrime)
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_xPrime, d))
    for i in range(num_of_xPrime):
        for j in range(d):
            cov_matrix[i,j] = cov_devY_devX(xPrime[i] ,value_at_max, model, l , j, m)
            
    return cov_matrix



#Compute the covariance between the derivative of x prime with respect ot m and the off diagonal hessian at the minimum
def cov_gradXprime_nonDiaHess(xPrime, value_at_max, model, l, m):
    num_of_xPrime = len(xPrime)
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_xPrime, d*(d-1)/2))
    
    for k in range(num_of_xPrime):
        index = 0
        for i in range(d):
            for j in range(i+1, d):
                cov_matrix[k, index] = cov_devdevY_devX(xPrime[k], value_at_max, model, l , i , j, m)
                index = index + 1
    
    return cov_matrix








#Compute the covariance between the derivative of x prime with respect ot m and the off diagonal hessian at the minimum
def cov_gradXprime_nonDiaHess(xPrime, value_at_max, model, l, m):
    num_of_xPrime = len(xPrime)
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_xPrime, d*(d-1)/2))
    
    for k in range(num_of_xPrime):
        index = 0
        for i in range(d):
            for j in range(i+1, d):
                cov_matrix[k, index] = cov_devdevY_devX(xPrime[k], value_at_max, model, l , i , j, m)
                index = index + 1
    
    return cov_matrix






#Compute the covariance between the derivative of x prime with respect ot m and the diagonal hessian at the minimum
def cov_gradXprime_diaHess(xPrime, value_at_max, model, l, m):
    num_of_xPrime = len(xPrime)
    d = len(value_at_max)
    cov_matrix = np.zeros((num_of_xPrime, d))
    
    for i in range(num_of_xPrime):
        for j in range(d):
            cov_matrix[i,j] = cov_devdevY_devX(xPrime[k], value_at_max, model, l , j , j, m)
            
    return cov_matrix


#Compute the covariance between the derivative of x prime with respect ot m and the minimum
def cov_gradXprime_max(xPrime, value_at_max, model, l, m):
    num_of_xPrime = len(xPrime)
    cov_matrix = np.zeros((num_of_xPrime, 1))
    for i in range(num_of_xPrime):
        cov_matrix[i, 0] = cov_devX_y(xPrime[i], value_at_max, model, l, m)
        
    return cov_matrix







#Compute the covariance between x and [c,z] with f(x_min) being the last element in [c, z]
def compute_cov_xPrime_cz(xPrime, nObservations, x_minimum, num_of_obser, model, noise, l_vec):
    x_nob = cov_xPrime_nObservations(xPrime, nObservations, model, l_vec)
    x_grad_min = cov_xPrime_maxGrad(xPrime, x_minimum, model, l_vec)
    x_off_dia = cov_xPrime_nonDiaHess(xPrime, x_minimum, model, l_vec)
    x_dia_hess = cov_xPrime_diaHess(xPrime, x_minimum, model, l_vec)
    x_x_min = cov_xPrime_max(xPrime, x_minimum, model, l_vec)

    result = np.concatenate((x_nob, x_grad_min, x_off_dia, x_dia_hess, x_x_min), axis = 1)
    # ? why only return result[0]
    # result = result[0]

    return result
    
    
    

#Compute the kernel matrix between z and z   
def compute_K_z(x_minimum, model, sigma, l_vec, noise, d):
    # sigma: https://docs.gpytorch.ai/en/stable/kernels.html#scalekernel
    # noise: https://docs.gpytorch.ai/en/latest/likelihoods.html#gaussianlikelihood 

    min_min = cov_max_max(x_minimum, model, noise, l_vec)
    dia_min = cov_diaHess_max(x_minimum, model, l_vec)
    dia_dia = cov_diaHess_diaHess(x_minimum, model, l_vec)
    
    first_row = np.concatenate((dia_dia, dia_min), axis = 1)
    second_row = np.concatenate((dia_min.T, min_min), axis = 1)
    
    result = np.vstack((first_row, second_row))
    result = result + sigma * (10**(-10))*np.eye(result.shape[0])
    
    return result






#Compute the kernel matrix between c and c
def compute_K_c(nObservations, x_minimum, num_of_obser, model, sigma, noise, λ, l_vec):
    
    d = len(x_minimum)
    x_minimum=x_minimum.reshape(-1,1)
    nob_nob = covNobeservations(nObservations, num_of_obser, model, noise, l_vec)
    nob_grad = cov_nObser_maxGrad(nObservations, x_minimum, num_of_obser, model, λ, l_vec)
    nob_off_dia = cov_nObser_off_maxHess(nObservations, x_minimum, num_of_obser, model,  λ,l_vec)
    grad_grad = cov_maxGrad_maxGrad(x_minimum, model, l_vec)
    grad_off_hess = cov_maxGrad_off_maxHess(x_minimum, model, l_vec)
    nonDia_nonDia = cov_nonDiaHess_nonDiaHess(x_minimum, model, l_vec)
    
    first_row = np.concatenate((np.concatenate((nob_nob, nob_grad), axis = 1), nob_off_dia), axis = 1 )
    second_row = np.concatenate((np.concatenate((nob_grad.T, grad_grad), axis = 1), grad_off_hess), axis = 1 )
    third_row = np.concatenate((np.concatenate((nob_off_dia.T, grad_off_hess.T), axis = 1), nonDia_nonDia), axis = 1 )
    
    result = np.vstack((first_row,second_row, third_row))
    result = result +  sigma * (10**(-10))*np.eye(result.shape[0])
    return result
    




    


    
#Compute cross covariance between c and z with last element being f(x_min)
def compute_K_cz(nObservations, x_minimum, num_of_obser, model, λ, noise, l_vec):
    d = len(x_minimum)
     
    #n Observations and diagonal hessian
    nob_dia_hess = cov_nObser_diagonal_maxHess(nObservations, x_minimum, num_of_obser, model, λ, l_vec)
        
    #n Observations and minimum
    nob_min = cov_nObser_max(nObservations, x_minimum, num_of_obser, model, λ, l_vec)
    
    #gradient at minimum and diagonal hessian
    grad_dia_hess = cov_maxGrad_diaHess(x_minimum, model, l_vec)
    
    #gradient at minimum and minimum
    grad_min  = cov_maxGrad_max(x_minimum, model, l_vec)
    
    #non diagonal hessian and diagonal hessian
    nonDia_dia = cov_nonDiaHess_diaHess(x_minimum, model, l_vec)
    
    #non diagonal hessian and minimum
    off_hess_min = np.array([cov_nonDiaHess_max(x_minimum, model, l_vec)]).T

        
    first_row = np.concatenate((nob_dia_hess, nob_min), axis = 1)
    second_row = np.concatenate((grad_dia_hess, grad_min), axis = 1)
    third_row = np.concatenate((nonDia_dia, off_hess_min), axis = 1)
    
    result = np.vstack((first_row, second_row, third_row))
    return result
    


    
#Use the results of K_Z, K_c and K_cz to compute the kernel matrix K. It is the same as K defined in the Appendix B.1
def compute_K(K_z, K_c, K_cz):
    first_row = np.concatenate((K_c, K_cz), axis = 1)
    second_row = np.concatenate((K_cz.T, K_z), axis = 1)
    result = np.vstack((first_row, second_row))
    return result





#Compute the kernel matrix of the n observations
def compute_KMM(Xsamples, model, noise, l_vec):
    m = len(Xsamples)
    Xbar = np.multiply(Xsamples, np.array(np.sqrt([l_vec,]*m)))
    Qbar = np.array([np.sum(np.multiply(Xbar,Xbar), axis = 1),]*m).T
    distance = Qbar + Qbar.T - 2*np.dot(Xbar, Xbar.T)
    result = model * np.exp(-0.5 * distance) + np.eye(m)*(noise + model*10**(-10))
    
    return result