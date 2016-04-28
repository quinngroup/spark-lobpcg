import argparse
import matplotlib.pyplot as plt
import numpy as np
from pcg import pcg
from pcg import lobpcg as mylobpcg
#from pyamg import smoothed_aggregation_solver
import scipy
import numpy
import scipy.linalg as sla
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as pairwise

from sklearn.metrics import silhouette_score
def laplacian(A, mode = 'norm'):
    """
    Constructs a normalized graph Laplacian from the given affinity matrix.

    Parameters
    ----------
    A : array, shape (N, N)
        Symmetric weighted affinity matrix.
    mode : string
        Specifies the norm of the resulting Laplacian. If "norm", uses the usual
        L = I - D^{-1/2} * A * D^{-1/2}. If anything else, uses our shortcut
        L = D^{-1/2} * A * D^{-1/2} which can have some advantages if the
        subsequent eigensolver only computes the leading eigenvalues/eigenvectors.

    Returns
    -------
    L : array, shape (N, N)
        Symmetric normalized graph Laplacian.
    """
    D = np.diag(1.0 / np.sqrt(A.sum(axis = 1)))
    if mode == 'norm':
        # "Traditional" normalized Laplacian.
        L = np.diag(np.ones(D.shape[0])) - D.dot(A).dot(D)
    elif mode =='D':
        return D
    else:
        # Our "shortcut" Laplacian.
        L = D.dot(A).dot(D)
    return L
def normalizeRows(U):
    #return np.array(map(lambda x: x / sla.norm(x), U))
	row_sums = U.sum(axis=1)
	return  U/row_sums[:, numpy.newaxis]

def eigens(L, k, mode = 'smallest'):
    """
    Performs the SVD/eigendecomposition and returns the k target eigenvectors
    and corresponding eigenvalues.

    Parameters
    ----------
    L : array, shape (N, N)
        Laplacian matrix.
    k : integer
        Number of principal components to return.
    mode : string
        Specifies whether to return the eigenvectors corresponding with the
        smallest eigenvalues, or largest.

    Returns
    -------
    w : array, shape (k,)
        Eigenvalues.
    V : array, shape (N, k)
        Principal components.
    """
    U, s, Vt = sla.svd(L, full_matrices = False)
    print("s[0]:", s[0])
    print("s[-1]:", s[-1])
    print("s[0]/s[-1]:",s[0]/s[-1])
    if mode == 'smallest':
        s = s[-k:]
        U = U[:, -k:]
    else:
        s = s[:k]
        U = U[:, :k]
    
    # Normalize the rows.
    #U = np.array(map(lambda x: x / sla.norm(x), U))
    return [s, normalizeRows(U)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Spectral Clustering",
        add_help = "How to use", prog = "spectralclustering.py")
    parser.add_argument("-i", "--input",
        help = "Path to input file.")

    # Optional arguments.
    parser.add_argument("-g", "--gamma", type = float, default = 0.05,
        help = "Value for gamma in building the RBF kernel. [DEFAULT: 0.05]")
    parser.add_argument("-k", "--clusters", type = int, default = 3,
        help = "Number of clusters. [DEFAULT: 3]")

    args = vars(parser.parse_args())
    g = args['gamma']
    k = args['clusters']

    # Read in the data.
    X = np.loadtxt(args['input'], delimiter = ",")

    # Construct a pairwise affinity matrix.
    A = pairwise.rbf_kernel(X, X, gamma = g)

    # Build the normalized graph Laplacian.
    
    L2 = laplacian(A,mode = "affinity") #D*A*D
    #L =  numpy.eye(L2.shape[0]) - L2 # I - D*A*D
    L = laplacian(A) #I - D*A*D
    #D = laplacian(A,mode='D')
    # Find the eigenvectors.
    #w,V = eigens(L, k)
    rX = numpy.random.rand(L.shape[1],k)
    #rX,_ = numpy.linalg.qr(rX)
    #n=L.shape[0]
    #M = numpy.diag(1./L.diagonal())
    #from pyamg import smoothed_aggregation_solver,rootnode_solver
    #M = smoothed_aggregation_solver(L2).aspreconditioner()
    
    #M=numpy.linalg.inv(L)
    M=None
    #M=numpy.random.rand(L.shape[0],L.shape[1])
    #M=M.T.dot(M)/2.
    #w,V = scipy.sparse.linalg.lobpcg(L,rX,largest=True,maxiter=1,M=M)
    w,V = mylobpcg(L,rX) #,largest=True,maxiter=1,M=M)
    #V=normalizeRows(V)
    #print(V)
    #V=V1
    #plt.scatter(range(V.shape[0]), sorted(V[:,13]), s = 200)
    #plt.show()
    # Cluster the eigenvectors.
    kmeans = cluster.KMeans(n_clusters = k)
    kmeans.fit(V)
    y = kmeans.labels_
    print(silhouette_score(X,y))
    # Plot the results.
    #plt.scatter(X[:, 0], X[:, 1], s = 200, c = y)
    #plt.show()
    
    from  sklearn.cluster import SpectralClustering as sksc
    sss = sksc(n_clusters=k,eigen_solver='arpack',gamma = g)
    y=sss.fit_predict(X)
    print(silhouette_score(X,y))
	