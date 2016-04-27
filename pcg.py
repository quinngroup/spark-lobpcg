
from scipy.sparse.linalg import lobpcg as slobpcg
def pcg(A,k,alpha =0.2, gamma =0.8):
    import scipy.linalg as sla
    import numpy
    i=0
    u=[]
    x=[]
    p=[]
    w=[]
    Ax = []
    for i in range(k):
        u.append(0)
        x.append(numpy.radnom.rand(A.shape[2],1))
        p.append(numpy.zeros((A.shape[2],1)))
        w.append(numpy.zeros((A.shape[2],1)))
        Ax.append(numpy.zeros((A.shape[2],1)))
    for i in range(k):
        Ax[i] = A.dot(x[i])
        u[i] = x[i].dot(x[i])/x[i].dot(Ax[i])
        w[i] = x[i] - u[i]*Ax[i]
    R = x.T.dot(A).dot(x)
    sla.eig(R)
	
def lobpcg(A,X,maxiter=20, largest=False, M=None, tol = 10e-6):
	import numpy, scipy
	k=X.shape[1]
	P = numpy.zeros(X.shape)
	pAp = numpy.zeros((X.shape[1],X.shape[1]))
	for i in range(maxiter):
		
		Ax= A.dot(X)
		xAx = X.T.dot(Ax)
		NewLambda = xAx.diagonal()/X.T.dot(X).diagonal()
		W = X.dot(numpy.diag(NewLambda)) - Ax
		if M is not None:
			W = M.dot(W)
		norm = numpy.linalg.norm(W)
		#print(i,norm)
		if norm < tol:
			return (NewLambda, X)
		
		#print(X.shape, W.shape, P.shape)
		RR = numpy.hstack([X,W,P])
		#RRR = RR.T.dot(A.dot(RR))
		#wAw = W.T.dot(A.dot(W))
		#print(xAx.shape,wAw.shape,pAp.shape)
		#RRRR = numpy.hstack([xAx,wAw,pAp])
		#pAp  = xAx
		#print(numpy.linalg.norm(RRR-RRRR))
		R = numpy.linalg.qr(RR)[0]
		#print(R.shape)
		P = X
		_, X = numpy.linalg.eigh(R.T.dot(A.dot(R)))
		#print(X.shape)
		if largest:
			X = R.dot(X[:,-k:])#numpy.linalg.qr(R.dot(X[:,:k]))[0]
		else:
			X = R.dot(X[:,:k])
		#X = X[:,:P.shape[1]]
	return NewLambda,X
		
if __name__=='__main__':
	import numpy
	n=3000
	k=3
	A = numpy.random.rand(n,n)
	A = (A.T*A)/2
	X = numpy.random.rand(n,k)
	S,V = lobpcg(A,X[:,:3],largest=True) #numpy.linalg.qr(X)[0][:,:k])
	S1,V1 = slobpcg(A,X[:,:3],largest=True) #numpy.linalg.qr(X)[0][:,:k])
	print(S)
	print(S1)
	print(numpy.linalg.eigh(A)[0][-k:])
	
	