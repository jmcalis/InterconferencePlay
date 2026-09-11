import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import log
import pandas as pd
import functions as fn


def RankZermelo(W, tol=10^-7, iterlimit = 10^4, powers = False):
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
    #pi is the strict power of each team
    ranking = np.argsort(pi) 
    #ranking is the ranking of each team (a 3 at index 10 means that team 10 is the 4th best team)
    if powers:
        return(pi)
    return(ranking)

def RankELO(W, initialELO):
    n = len(W[0,:])

    k = log(10)/400  # Logistic function steepness parameter
    update_k = 90  # Elo update factor
    season = list()
    for i in range(0,n):
        for j in range (i,n):
            if W[i,j]>0 or W[j,i]>0:
                if W[i,j]>W[j,i]:
                    season.append([i,j,0])
                elif W[j,i]>W[i,j]: 
                    season.append([j,i,0])
                else:
                    season.append([i,j,1])
    season = np.random.permutation(season)
    df = pd.DataFrame(season,columns=["W","L","D"])

    ratings = initialELO
    for index, row in df.iterrows():
        winner = row['W']
        loser = row['L']
        winner_elo = ratings[winner]
        loser_elo = ratings[loser]

        ## Home-field advantage adjustment
        #if row['Location'] == 'Home':
        #    h = 55  # Home-field advantage for the winner
        #elif row['Location'] == 'Away':
        #    h = -55  # Home-field disadvantage for the winner
        #else:
        #    h = 0  # Neutral field, no advantage

        winner_probability = fn.prob_win(winner_elo, loser_elo, k, 0)

        #winner_probabilities.append(winner_probability)
        #winner_pregame_elos.append(winner_elo)
        #loser_pregame_elos.append(loser_elo)

        # Update Elo ratings based on the game outcome
        updated_winner_elo, updated_loser_elo = fn.update_elo(winner_elo, loser_elo, k= k, K=update_k, outcome=1, h=0)

        # Store the updated Elo ratings back in the dictionary
        ratings[winner] = updated_winner_elo
        ratings[loser] = updated_loser_elo

    return(np.argsort(ratings))
    
def simulate(n,m,M, g=9, dist = 'uniform', method = 'random', bias = 0):
    #Start with a ground truth about N = n+m teams
    if (dist == 'uniform'):
        pi = np.random.rand(n+m)
    else:
        pi = np.random.exponential(1,m+n)

    b = np.concatenate((bias*np.ones(n),np.zeros(m)))
    pi = pi+b
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

def scanZer(n,m,Mmin,Mmax,reps = 100,g=9, dist = 'uniform',method = 'random', bias = 0):
    Ms = range(Mmin,Mmax)
    df = pd.DataFrame(columns=["M","b","LD","CW","CT4","CT12"])
    for M in range(Mmin,Mmax):
        for k in range(0,reps):
            result = simulate(m,n,M, g=g,dist=dist,method=method,bias = bias)
            W=result[0]
            pi = result[1]
            tp = np.argsort(np.argsort(pi))
            foundpi = RankZermelo(W)
            fp = np.argsort(foundpi)
            newrow = pd.DataFrame({"M":M,"b":bias,"LD": L1dist(tp,fp),"CW": correct_winner(tp,fp), "CT4": correct_top_four(tp,fp),"CT12":correct_top_12(tp,fp)},index = [0])
            df=pd.concat([df,newrow])
    return(df)

def scanELO(n,m,Mmin,Mmax,reps = 100,g=9,dist = 'uniform', method = 'random', bias = 0, info=0.5,spread = 100):
    df = pd.DataFrame(columns=["M","b","LD","CW","CT4","CT12"])
    for M in range(Mmin,Mmax):
        for k in range(0,reps):
            result = simulate(m,n,M, g=g,dist=dist,method=method,bias = bias)
            W=result[0]
            pi = result[1]
            initialELO = 1500*pi/np.mean(pi)*(1-info)+info*np.random.normal(1500,spread,n+m)
            tp = np.argsort(np.argsort(pi))
            foundpi = RankELO(W,initialELO)
            fp = np.argsort(foundpi)
            newrow = pd.DataFrame({"M":M,"b":bias,"LD": L1dist(tp,fp),"CW": correct_winner(tp,fp), "CT4": correct_top_four(tp,fp),"CT12":correct_top_12(tp,fp)},index = [0])
            df=pd.concat([df,newrow])
    return(df)

def correct_winner(ptrue,pfound):
    num = np.argmin(ptrue)-np.argmin(pfound)
    return num==0

def correct_top_four(ptrue,pfound):
    num0 = abs(np.argmin(ptrue)-np.argmin(pfound))
    num1 = abs(np.where(ptrue==1)[0]-np.where(pfound==1)[0])
    num2 = abs(np.where(ptrue==2)[0]-np.where(pfound==2)[0])
    num3 = abs(np.where(ptrue==3)[0]-np.where(pfound==3)[0])
    ret = num0+num1+num2+num3==0
    return (ret[0])

def correct_top_12(ptrue,pfound):
    sum = 0
    for i in range(0,12):
        sum = sum+abs(np.where(ptrue==i)[0]-np.where(pfound==i)[0])
    ret = sum==0
    return(ret[0])

def L1dist(ptrue,pfound):
    return(sum(abs(ptrue-pfound))/len(ptrue))

def rankDistance(rfound,n,m,t=-1):
    rAfound =np.sort(rfound[0:n])
    rBfound=np.sort(rfound[n:-1])
    
    rAfound = rAfound[0:t]
    rBfound = rBfound[0:t]
    
    dif = np.mean(rAfound)-np.mean(rBfound)
    return(dif)

def scoreDistance(pifound,n,m,t=-1):
    piAfound =np.sort(pifound[0:n])
    piBfound=np.sort(pifound[n:-1])

    piAfound = piAfound[0:t]
    piBfound = piBfound[0:t]

    dif = np.mean(piAfound)-np.mean(piBfound)
    return(dif)

def difDistribution(n,m,M,t=-1,reps=2000, g=9,dist = 'uniform', method = 'random', bias=0):
    d = np.zeros(reps)
    for i in range(reps):
        result = simulate(n,m,M,g,dist,method,bias)
        W=result[0]
        pi = RankZermelo(W,powers=True)
        d[i]=scoreDistance(pi,n,m,t)
    return(d)

def rankDifDistribution(n,m,M,t=-1,reps=2000, g=9,dist = 'uniform', method = 'random', bias=0):
    d = np.zeros(reps)
    for i in range(reps):
        result = simulate(n,m,M,g,dist,method,bias)
        W=result[0]
        r = RankZermelo(W,powers=False)
        d[i]=rankDistance(r,n,m,t)
    return(d)

def get_ecdf(data):
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return([data_sorted,p])

