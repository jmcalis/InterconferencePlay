import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def Rank(W, tol=10^-7, iterlimit = 10^4):
    n = len(W[0,:])
    pi = np.random.rand(n)
    finished = False
    iter = 0
    while (not finished) and iter<iterlimit :
        pinew = np.zeros(n)
        pinew[np.isinf(pi)]=np.inf
        pimat = np.tile(pi,(n,1))
        pimat = pimat + pimat.transpose() #Get matrix of pi_i+pi_j
        nummat = np.divide(W,pimat) # get matrix of w_{i,j}/(pi_i+pi_j)
        denommat = np.divide(W.transpose(),pimat) #get matrix of w_{j,i}/(pi_1+pi_j)

        num = np.matmul(nummat,pi.transpose())
        denom = np.matmul(denommat,np.ones(n))
        pinew = np.divide(num,denom)
        if sum(abs(pinew-pi))<tol:
            finished = True
        pi = pinew 
        iter = iter+1
    ranking = np.argsort(pi)
    return(ranking)

def simulate(n,m,M,reps = 1, g=9, dist = 'uniform', method = 'random'):
    #Start with a ground truth about N = n+m teams
    if (dist == 'uniform'):
        pi = np.random.rand(n+m)
    else:
        pi = np.random.exponential(1,m+n)
    #Create a schedule with 9 conference games each and M non-conference games (drawn from some distribution)
    G1 = nx.random_regular_graph(g,n)
    G2 = nx.random_regular_graph(g,m)
    G = nx.disjoint_union(G1,G2)

    for i in range(0,M):
        r1 = np.random.randint(0,n)
        r2 = np.random.randint(0,m)
        G.add_edge(r1,n+r2)

    W=nx.adjacency_matrix(G).toarray()

    #use ground truth to predict scores form each match

    pmat = np.tile(pi,(n+m,1))
    pmat = np.divide(pmat.transpose(),pmat+pmat.transpose())
    scores = np.random.binomial(40, pmat)
    scores = np.multiply(W,scores)
    return(scores, pi)

def scan(n,m,Mmin,Mmax,reps = 100,g=9, dist = 'uniform',method = 'random'):
    Ms = range(Mmin,Mmax)

    LDmeans=np.zeros(Mmax-Mmin)
    CWmeans=np.zeros(Mmax-Mmin)
    CT4means=np.zeros(Mmax-Mmin)
    CT12means=np.zeros(Mmax-Mmin)
    Miter = 0
    for M in range(Mmin,Mmax):
        LDs=np.zeros(reps)
        CW=np.zeros(reps)
        CT4=np.zeros(reps)
        CT12=np.zeros(reps)
        kiter = 0
        for k in range(0,reps):
            result = simulate(m,n,M,reps =1, g=g,dist=dist,method=method)
            W=result[0]
            pi = result[1]
            tp = np.argsort(np.argsort(pi))
            foundpi = Rank(W)
            fp = np.argsort(foundpi)
            LDs[kiter]=L1dist(tp,fp)
            CW[kiter]=correct_winner(tp,fp)
            CT4[kiter]=correct_top_four(tp,fp)
            CT12[kiter]=correct_top_12(tp,fp)
            kiter=kiter+1
        LDmeans[Miter]=np.mean(LDs)
        CWmeans[Miter]=np.mean(CW)
        CT4means[Miter]=np.mean(CT4)
        CT12means[Miter]=np.mean(CT12)
        Miter = Miter+1

    return(Ms,LDmeans,CWmeans,CT4means,CT12means)

def correct_winner(ptrue,pfound):
    num = np.argmin(ptrue)-np.argmin(pfound)
    return num==0

def correct_top_four(ptrue,pfound):
    num0 = abs(np.argmin(ptrue)-np.argmin(pfound))
    num1 = abs(np.where(ptrue==1)[0]-np.where(pfound==1)[0])
    num2 = abs(np.where(ptrue==2)[0]-np.where(pfound==2)[0])
    num3 = abs(np.where(ptrue==3)[0]-np.where(pfound==3)[0])
    return(num0+num1+num2+num3==0)

def correct_top_12(ptrue,pfound):
    sum = 0
    for i in range(0,12):
        sum = sum+abs(np.where(ptrue==i)[0]-np.where(pfound==i)[0])
    return(sum==0)

def L1dist(ptrue,pfound):
    return(sum(abs(ptrue-pfound))/len(ptrue))


