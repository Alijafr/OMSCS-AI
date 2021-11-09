
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: solution.ipynb

import numpy as np
from helper_functions import *

def get_initial_means(array, k):
    """
    Picks k random points from the 2D array
    (without replacement) to use as initial
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """
    initial_means = np.random.choice(array.shape[0],k,replace=False)

    return array[initial_means]

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and
    calculate new means.
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """
    diffs = []
    clusters = []
    new_means = []
    #short for loop to take the square distance between the means and all other points
    for i in range(len(means)):
        diff = np.linalg.norm(X - means[i],axis=1)
        diffs.append(diff)

    #convert the differences into np array to take advantage of the vectorization
    diffs_np = np.array(diffs)
    #find what cluster each point belong to
    cluster_index = np.argmin(diffs_np,axis=0)

    #another short loop to seperate the points into clusters and find their means
    for i in range(len(means)):
        cluster = X[cluster_index==i]
        mean = np.mean(cluster,axis=0)
        new_means.append(mean)

    return np.array(new_means),cluster_index

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """
    r,c,ch = image_values.shape
    image_values_reshaped = image_values.reshape(-1,3)
    m,n = image_values_reshaped.shape
    if initial_means is None:
        means = get_initial_means(image_values_reshaped,k)
    else:
        means = initial_means
    clusters = np.zeros([m])
    while True:
        new_means, clusters = k_means_step(image_values_reshaped, k, means)
        if (new_means == means).all():
            break
        means = new_means
    #reshape the clusters
    clusters = clusters.reshape(r,c)

    #update the image
    image = np.copy(image_values)
    for i in range(k):
        image [ clusters==i] = means[i]

    return image




########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def initialize_parameters(X, k, MU=None):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    """
    m,n = X.shape
    if MU is None:
        index_of_means = np.random.choice(X.shape[0],k,replace=False)
        MU = X[index_of_means]

    #calculate sigma
    #formala => cov (sigma) = sum((xi-mu).T * (xi-mu)) --> this is the definition of a matrix multiplication
    num_samples = len(X)
    sigma = np.zeros((k,n,n))
    for i in range(k):
        diff = X - MU[i]
        sigma_ = (diff.T @ diff)/num_samples
        sigma[i,:,:]= sigma_

    #calculate PI
    PI = np.zeros((k,1))
    PI[:] = 1/k

    return MU,sigma,PI

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] or numpy.ndarray[numpy.ndarray[float]]
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float or numpy.ndarray[float]
    """
    d = len(sigma)
    diff = x-mu
    sigma_det = np.linalg.det(sigma)
    normalizer =np.sqrt(sigma_det)*(2*np.pi)**(d/2)
    if x.ndim == 1:
        p = np.exp(-0.5*diff@np.linalg.inv(sigma)@diff.T)
        p /= normalizer
    else:
        A = diff@np.linalg.inv(sigma)
        #now i need to only multiply row i with i cloumn, the other multipication is irrelevant
        B = np.einsum('ij,ji->i',A,diff.T)
        p = np.exp(-0.5*B)/normalizer

        ##NOTE: implementation below is not efficient

#         #in case, of many sample together, we can use this trick
#         p = np.exp(-0.5*diff@np.linalg.inv(sigma)@diff.T) #this will produce mxm matrix but we want 1xm matrix
#         p /= normalizer
#         #it turns out, the only relevant caluclations are the diag of p, the rest are meaningless
#         p = np.diag(p)

    return p

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    m,n = X.shape
    res = np.zeros((k,m))
    for i in range(k):
        res[i,:] = PI[i]*prob(X,MU[i],SIGMA[i])

    #normalize
    normalizer = np.sum(res,axis=0)
    res /=normalizer

    return res


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    """
    m,n = X.shape
    mc = np.sum(r,axis=1)

    #new PI
    new_PI = mc /np.sum(mc)

    #new mu
    new_MU = np.zeros((k,n))
    for i in range(k):
        new_MU[i,:] = np.sum(np.multiply(r[i,np.newaxis].T,X),axis=0)/mc[i]

    m, n = X.shape
    new_SIGMA = np.zeros((k,n,n))
    for i in range(k):
        diff = X - new_MU[i]
        sigma_ = (r[i,np.newaxis]*diff.T@diff)/mc[i]
        new_SIGMA[i,:,:]= sigma_

    return (new_MU, new_SIGMA, new_PI)




########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((n=1 to N), log(sum((k=1 to K),
                                      mixing_k * N(x_n | mean_k,stdev_k))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    returns:
    log_likelihood = float
    """
    m,n = X.shape
    ps = np.zeros((k,m))
    for i in range(k):
        ps[i,:] = PI[i]*prob(X,MU[i],SIGMA[i])
    log_likelihood = np.log(np.sum(ps,axis=0))
    log_likelihood = np.sum(log_likelihood)

    return log_likelihood


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        MU,SIGMA, PI = initialize_parameters(X,k)
    else:
        MU,SIGMA, PI = initial_values
    convergence = False
    counter = 0
    pre_likelihood = likelihood(X,PI,MU,SIGMA,k)
    while not convergence:
        r = E_step(X,MU,SIGMA,PI,k)
        MU, SIGMA, PI = M_step(X,r,k)
        current_likelihood = likelihood(X,PI,MU,SIGMA,k)
        counter, convergence = convergence_function(pre_likelihood,current_likelihood,counter)
        pre_likelihood = current_likelihood
    return (MU, SIGMA, PI, r)



########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood
    (maximum responsibility value).

    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    return:
    clusters = numpy.ndarray[int] - m x 1
    """
    cluster = np.argmax(r,axis=0)
    return cluster

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def segment(X, MU, k, r):
    """
    Segment the X matrix into k components.
    Returns a matrix where each data point is
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood
    component mean. (the shape is still mxn, not
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """
    new_X = np.copy(X)
    clusters = cluster(r)
    for i in range(k):
        new_X[ clusters==i] = MU[i]

    return new_X

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    m,n = X.shape
    new_X = np.copy(X)
    best_likelihood = float('-inf')
    for i in range(iters):
        MU, SIGMA, PI, r =train_model(X,k,default_convergence)
        current_likelihood = likelihood(X,PI,MU,SIGMA,k)
        if current_likelihood > best_likelihood:
            best_likelihood = current_likelihood
            new_X = segment(X,MU,k,r)


    return best_likelihood, new_X


def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    
    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1 
    """
    MU, SIGMA, PI, r =train_model(X,k,default_convergence)
    
    PI = np.zeros((k,1))
    PI[:] = 1/k 
    
    return (MU, SIGMA, PI)
def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(previous_variables[0]-new_variables[0]) < 0.1 *abs(previous_variables[0])).all() and \
                                (abs(previous_variables[1]-new_variables[1]) < 0.1 *abs(previous_variables[1])).all() and \
                                (abs(previous_variables[2]-new_variables[2]) < 0.1 *abs(previous_variables[2])).all()

    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0
    
    return conv_ctr, conv_ctr > conv_ctr_cap

def train_model_improved(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the 
    expectation-maximization algorithm. 
    E.g., iterate E and M steps from 
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction 
    implemented above. 

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k x 1
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    if initial_values is None:
        MU,SIGMA, PI = improved_initialization(X,k)
    else:
        MU,SIGMA, PI = initial_values
    convergence = False
    counter = 0
    previous_variables = (MU, SIGMA, PI) 
    while not convergence:
        r = E_step(X,MU,SIGMA,PI,k)
        MU, SIGMA, PI = M_step(X,r,k)
        new_variables = (MU, SIGMA, PI)
        current_likelihood = likelihood(X,PI,MU,SIGMA,k)
        counter, convergence = convergence_function(previous_variables,new_variables,counter)
        previous_variables = new_variables
    return (MU, SIGMA, PI, r)

def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k x 1
    k = int

    return:
    bayes_info_criterion = int
    """
    m,n = X.shape
    num_params = n*k + (k * n * (n + 1) / 2) + k

    BIC = num_params*np.log(m) - 2*likelihood(X,PI,MU,SIGMA,k)

    return BIC

def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC 
    and maximum likelihood with respect
    to image_matrix and comp_means.
    
    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    m,n = image_matrix.shape
    best_likelihood = float('-inf')
    best_BIC = float('inf')
    n_comp_min_bic = None
    n_comp_max_likelihood = None
    for i, means in enumerate(comp_means):
        k = len(means)
        initial_values = initialize_parameters(image_matrix,k,means)
        MU, SIGMA, PI, r =train_model(image_matrix,k,default_convergence,initial_values=initial_values)
        current_likelihood = likelihood(image_matrix,PI,MU,SIGMA,k)
        current_BIC = bayes_info_criterion(image_matrix,PI,MU,SIGMA,k)
        if current_likelihood > best_likelihood:
            best_likelihood = current_likelihood
            n_comp_max_likelihood = k
        if current_BIC < best_BIC:
            best_BIC = current_BIC
            n_comp_min_bic = k
    
    return (n_comp_min_bic, n_comp_max_likelihood)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################




def return_your_name():
    # return your name
    # TODO: finish this
    return "Ali Alrasheed"