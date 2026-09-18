import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import log
import pandas as pd
import functions as fn
import seaborn as sns


def RankZermelo(W, tol=10**-7, iterlimit = 10**4, powers = False):
    '''
    Computes the powers and ranks for each team according to the pairwise comparison data in matrix form

    Parameters
    ----------
    W : numpy array 
        A square matrix with pairwise comparison data
    tol : float, optional
        The tolerence required by the interative method to hault
    iterlimit: int, optional
        Maximum number of interations used in the iterative method
    powers: bool, optional
        If the method returns the powers of each team. If it is False, the method returns the rank of each team
    Returns 
    ----------
    rankings : numpy array
        A ranked list of teams (a 3 at index 10 means that team 3 is the 11th best team) or a list of powers
    '''
    #Initialize iterative method components
    n =len(W[0,:])
    pi = np.random.rand(n)
    finished = False
    iter = 0
    while (not finished) and iter<iterlimit :
        #carry out iterative method synchronously (see Newman 2024 for details)
        pinew = np.zeros(n) #Pre allocate space
        pinew[np.isinf(pi)]=np.inf #precondition in the case that a single team has never lost
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
    #pi is the power of each team
    ranking = np.argsort(pi) 
    #ranking is the ranked list each teams (a 3 at index 10 means that team 3 is the 11th best team)
    if powers:
        return(1/max(pi)*pi)
    return(ranking)

def RankELO(W, initialELO,gameData = True, returnRat = False):
    '''
    Computes the ELO rating and ranks for each team according to the pairwise comparison data in matrix form

    Parameters
    ----------
    W : numpy array 
        A square matrix with pairwise comparison data
    initialELO: numpy array 
        An array of Elo ratings the same length as the number of teams
    gameData: bool, optional
        If true, the method uses only win-loss data from W. If false, the method will use score data (extremely slowly)
    returnRat: bool, optional
        If the method returns the Elo ratings of each team. If it is False, the method returns the rank of each team
    
    Returns
    ----------
    rankings : numpy array
            A ranked list of teams (a 3 at index 10 means that team 3 is the 11th best team) or a list of Elo ratings
    '''
    n = len(W[0,:])

    k = log(10)/400  # Logistic function steepness parameter
    update_k = 90  # Elo update factor
    season = list()
    if gameData:
        for i in range(0,n):
            for j in range (i,n):
                if W[i,j]>0 or W[j,i]>0:
                    if W[i,j]>W[j,i]:
                        season.append([i,j,0])
                    elif W[j,i]>W[i,j]: 
                        season.append([j,i,0])
                    else:
                        season.append([i,j,1])
    else:
        for i in range(0,n):
            for j in range (0,n):
                if W[i,j]>0:
                    season.extend([[i,j,0]]*W[i,j])

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
    if returnRat:
        return (ratings)
    else:
        return(np.argsort(ratings))
    
def simulate(n,m,M, g=9, dist = 'uniform', bias = 0):
    '''
    Simulates results from a league with two conferences with few interconference games

    Parameters
    ----------
    n : int 
        number of teams in conference A
    m: int
        number of teams in conference B
    M: int
        number of interconference games total
    g: int, optional
        number of conference games played by each team
    dist: string, optional
        If dist is "uniform" powers of each team are selected uniformly. Otherwise they are selected exponentially
    bias: float, optional
        A float which is added to the randomly generated powers of teams in conference A but not B

    Returns
    ----------
    W : numpy array
        A square matrix containing score data
    pi: numpy array
        A list of team powers    
    '''

    #Start with a ground truth about N = n+m teams
    if (dist == 'uniform'):
        pi = np.random.rand(n+m)
    else:
        pi = np.random.exponential(1,m+n)

    #Add bias
    b = np.concatenate((bias*np.ones(n),np.zeros(m)))
    pi = pi+b
    #normalize because power is only unique up to a proportionality factor
    pi = 1/max(pi)*pi
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
    temp = np.maximum(np.random.normal(57,16,(n+m,n+m)),3) #57 and 16 estimated from real CFB data

    pointtotal =np.array(np.triu(temp,1)+np.triu(temp,1).transpose(),dtype=int)
    scores = np.random.binomial(pointtotal, pmat)#average total points per game 
    scores = np.multiply(W,scores)
    return(scores, pi)

def simulateEnforced(n,m,M, g=9, dist = 'uniform', comparison = False):
    '''
    Simulates results from a league with two conferences with few interconference games 
    in which both conferences have equal average power

    Parameters
    ----------
    n : int 
        number of teams in conference A
    m: int
        number of teams in conference B
    M: int
        number of interconference games total
    g: int, optional
        number of conference games played by each team
    dist: string, optional
        If dist is "uniform" powers of each team are selected uniformly. Otherwise they are selected exponentially
    comparison: bool, optional
        A flag to use this method to add M conference games to the schedule instead of M interconference games
    
    Returns
    ----------
    W : numpy array
        A square matrix containing score data
    pi: numpy array
        A list of team powers    
    '''
    #Start with a ground truth about N = n+m teams
    if (dist == 'uniform'):
        pi = np.random.rand(n+m)
    else:
        pi = np.random.exponential(1,m+n)

    #enforce that the conferences have the same mean power
    cA=pi[0:n]
    cB=pi[n:]
    meanA = np.mean(cA)
    meanB = np.mean(cB)
    diff = meanA-meanB 
    if diff<0:
        pi = pi +np.concatenate((-diff*np.ones(n),np.zeros(m)))
    else:
        pi = pi +np.concatenate((np.zeros(n),diff*np.ones(m)))


    #Create a schedule with 9 conference games each and M non-conference games (drawn from some distribution)
    G1 = nx.random_regular_graph(g,n)
    G2 = nx.random_regular_graph(g,m)
    G = nx.disjoint_union(G1,G2)
    if comparison:
        for i in range(0,M):
            A = np.random.rand()<n/(n+m)
            if A:
                edge=np.random.choice(n,2)
                G.add_edge(edge[0],edge[1])
            else:
                edge=np.random.choice(range(n,n+m),2)
                G.add_edge(edge[0],edge[1])

    else:
        for i in range(0,M):
            r1 = np.random.randint(0,n)
            r2 = np.random.randint(0,m)
            G.add_edge(r1,n+r2)

    W=nx.adjacency_matrix(G).toarray()

    #use ground truth to predict scores form each match
    pmat = np.tile(pi,(n+m,1))
    pmat = np.divide(pmat.transpose(),pmat+pmat.transpose())
    temp = np.maximum(np.random.normal(57,16,(n+m,n+m)),3)
    pointtotal =np.array(np.triu(temp,1)+np.triu(temp,1).transpose(),dtype=int)
    scores = np.random.binomial(pointtotal, pmat)#average total points per game 
    scores = np.multiply(W,scores)

    return(scores, pi)

def scanZer(n,m,Mmin,Mmax,reps = 100,g=9, dist = 'uniform', bias = 0):
    '''
    Simmulates many season of a league with two conferences with few interconference games 
    and performs the Zermelo ranking algorithm 

    Parameters
    ----------
    n : int 
        number of teams in conference A
    m: int
        number of teams in conference B
    Mmin: int
        minimum number of interconference games total
    Mmax: int
        maximum number of interconference games total
    g: int, optional
        number of conference games played by each team
    dist: string, optional
        If dist is "uniform" powers of each team are selected uniformly. Otherwise they are selected exponentially
    bias: float, optional
        A float which is added to the randomly generated powers of teams in conference A but not B
    
    Returns
    ----------
    df: pandas DataFrame
        A dataframe containing the results of each simulated season   
    '''
    Ms = range(Mmin,Mmax)
    df = pd.DataFrame(columns=["M","b","LD","CW","CT4","CT12"])
    for M in range(Mmin,Mmax):
        for k in range(0,reps):
            #simulate a season then perform the rankings
            result = simulate(m,n,M, g=g,dist=dist,bias = bias)
            W=result[0]
            pi = result[1]
            tp = np.argsort(np.flip(np.argsort(pi)))#tp is a list of rankings (If 3 is in index 10, team 10 is ranked 4th)
            foundpi = RankZermelo(W)
            fp = np.argsort(np.flip(foundpi))#fp is a list of rankings (If 3 is in index 10, team 10 is ranked 4th)
            #fill dataframe with results
            newrow = pd.DataFrame({"M":M,"b":bias,"LD": L1dist(tp,fp),"CW": correct_winner(tp,fp), "CT4": correct_top_four(tp,fp),"CT12":correct_top_12(tp,fp)},index = [0])
            df=pd.concat([df,newrow])
    return(df)

def scanELO(n,m,Mmin,Mmax,reps = 100,g=9,dist = 'uniform', method = 'random', bias = 0, info=0.5,spread = 100,gamedata=True):
    '''
    Simmulates many season of a league with two conferences with few interconference games 
    and performs the Zermelo ranking algorithm 

    Parameters
    ----------
    n : int 
        number of teams in conference A
    m: int
        number of teams in conference B
    Mmin: int
        minimum number of interconference games total
    Mmax: int
        maximum number of interconference games total
    g: int, optional
        number of conference games played by each team
    dist: string, optional
        If dist is "uniform" powers of each team are selected uniformly. Otherwise they are selected exponentially
    bias: float, optional
        A float which is added to the randomly generated powers of teams in conference A but not B
    info: float, optional
        The fidelity of the initial Elo ratings to the ratings used in simulation (1 is absolutely acurate)
    spread: float, optional
        The standard deviation in the noise added to the initial Elo ratings
    gamedata: bool, optional
        If true, the rankELO method uses only win-loss data for every simulated season. If False it uses score data (very slowly)
    
    Returns
    ----------
    df: pandas DataFrame
        A dataframe containing the results of each simulated season   
    '''
    df = pd.DataFrame(columns=["M","b","LD","CW","CT4","CT12","info","spread"])
    for M in range(Mmin,Mmax):
        for k in range(0,reps):
            result = simulate(m,n,M, g=g,dist=dist,method=method,bias = bias)
            W=result[0]
            pi = result[1]
            initialELO = (info*(np.log(pi)*100+1500))+(1-info)*np.random.normal(1500,spread,n+m)
            tp = np.argsort(np.flip(np.argsort(pi))) #tp is a list of rankings (If 3 is in index 10, team 10 is ranked 4th)
            foundpi = RankELO(W,initialELO,gamedata)
            fp = np.argsort(np.flip(foundpi)) #fp is a list of rankings (If 3 is in index 10 team 10 is ranked 4th)
            newrow = pd.DataFrame({"M":M,"b":bias,"LD": L1dist(tp,fp),"CW": correct_winner(tp,fp), "CT4": correct_top_four(tp,fp),"CT12":correct_top_12(tp,fp), "info":info, "spread":spread},index = [0])
            df=pd.concat([df,newrow])
    return(df)

def correct_winner(ptrue,pfound):
    '''
    Determines if the two rankings have the same best ranked team

    Parameters
    ----------
    ptrue: numpy array
        A list of rankings or powers
    pfound: numpy array
        A list of rankings of powers
    
    Returns
    ---------
    ret: bool
        True if the two lists have the same best ranked team
    '''
    num = np.argmin(ptrue)-np.argmin(pfound)
    return num==0

def correct_top_four(ptrue,pfound):
    '''
    Determines if the two rankings have the same top 4 in the same order

    Parameters
    ----------
    ptrue: numpy array
        A list of rankings
    pfound: numpy array
        A list of rankings
            
    Returns
    ---------
    ret: bool
        True if the two lists have the same top 4
    '''
    num0 = abs(np.argmin(ptrue)-np.argmin(pfound))
    num1 = abs(np.where(ptrue==1)[0]-np.where(pfound==1)[0])
    num2 = abs(np.where(ptrue==2)[0]-np.where(pfound==2)[0])
    num3 = abs(np.where(ptrue==3)[0]-np.where(pfound==3)[0])
    ret = num0+num1+num2+num3==0
    return (ret[0])

def correct_top_12(ptrue,pfound):
    '''
    Determines if the two rankings have the same top 12 in the same order

    Parameters
    ----------
    ptrue: numpy array
        A list of rankings
    pfound: numpy array
        A list of rankings
    '''
    sum = 0
    for i in range(0,12):
        sum = sum+abs(np.where(ptrue==i)[0]-np.where(pfound==i)[0])
    ret = sum==0
    return(ret[0])

def L1dist(ptrue,pfound):
    '''
    Determines the distance between two rankings in the L1 sense

    Parameters
    ----------
    ptrue: numpy array
        A list of rankings
    pfound: numpy array
        A list of rankings
            
    Returns
    ---------
    ret: float
        L1 distance between the two ranked lists
    '''
    return(sum(abs(ptrue-pfound))/len(ptrue))

def rankDistanceComp(rfound,rtrue, n,m,t=None):
    '''
    Determines the difference in average rank between two conferences in found ranking relative to true ranking

    Parameters
    ----------
    rfound: numpy array
        A ranked list of teams
    rtrue: numpy array
        A ranked list of teams
    n: int
        size of confernce A
    m: int
        size of conference B
    t: int optional
        The number of teams measured from the top 
            
    Returns
    ---------
    dif: float
        the difference between the measured difference between conference ranks and the true difference between conference ranks
    '''
    rAfound =np.sort(rfound[0:n])
    rBfound=np.sort(rfound[n:])
    rAtrue =np.sort(rtrue[0:n])
    rBtrue=np.sort(rtrue[n:])
    if t is not None:
        rAfound = rAfound[:t]
        rBfound = rBfound[:t]
        rAtrue = rAtrue[:t]
        rBtrue = rBtrue[:t]
        
    dif = (np.mean(rAfound)-np.mean(rBfound))-(np.mean(rAtrue)-np.mean(rBtrue))
    return(dif)

def rankDistance(rfound,n,m,t=None):
    '''
    Determines the difference in average rank between two conferences in found ranking

    Parameters
    ----------
    rfound: numpy array
        A ranked list of teams
    n: int
        size of confernce A
    m: int
        size of conference B
    t: int optional
        The number of teams measured from the top 
            
    Returns
    ---------
    dif: float
         measured difference between conference ranks 
    '''

    rAfound =np.sort(rfound[0:n])
    rBfound=np.sort(rfound[n:])
    if t is not None:
        rAfound = rAfound[:t]
        rBfound = rBfound[:t]
    
    dif = np.mean(rAfound)-np.mean(rBfound)
    return(dif)

def scoreDistanceComp(pifound,pitrue, n,m,t=None):
    '''
    Determines the difference in average power between two conferences in found power relative to true power

    Parameters
    ----------
    pifound: numpy array
        A list of team powers
    pitrue: numpy array
        A list of team powers
    n: int
        size of confernce A
    m: int
        size of conference B
    t: int optional
        The number of teams measured from the top 
            
    Returns
    ---------
    dif: float
        the difference between the measured difference between conference powers and the true difference between conference powers
    '''

    piAfound =np.sort(pifound[0:n])
    piBfound=np.sort(pifound[n:])
    piAtrue =np.sort(pitrue[0:n])
    piBtrue=np.sort(pitrue[n:])

    if t is not None:
        piAfound = piAfound[-t:]
        piBfound = piBfound[-t:]
        piAtrue = piAtrue[-t:]
        piBtrue = piBtrue[-t:]
        
    dif = (np.mean(piAfound)-np.mean(piBfound))-(np.mean(piAtrue)-np.mean(piBtrue))
    return(dif)

def scoreDistance(pifound,n,m,t=None):
    '''
    Determines the difference in average power between two conferences in found power

    Parameters
    ----------
    pifound: numpy array
        A list of team powers
    n: int
        size of confernce A
    m: int
        size of conference B
    t: int optional
        The number of teams measured from the top 
            
    Returns
    ---------
    dif: float
        the measured difference between conference powers
    '''
    piAfound =np.sort(pifound[0:n])
    piBfound=np.sort(pifound[n:])

    if t is not None:
        piAfound = piAfound[-t:]
        piBfound = piBfound[-t:]

    dif = np.mean(piAfound)-np.mean(piBfound)
    return(dif)

def difDistribution(n,m,M,t=None,reps=2000, g=9,dist = 'uniform', bias=0,comp = False):
    '''
    Finds an emperical distribution of observed differences between average conference powers

    Parameters
    ----------
    n: int
        size of confernce A
    m: int
        size of conference B
    M: int
        Total number of interconference games
    t: int optional
        The number of teams measured from the top
    reps: int optional
        Number of simulated seasons measured
    g: int optional
        Number of conference games 
    dist: string optional
        If "uniform" the true powers are generated uniformly. If not they are generated exponentially.
    bias: float, optional
        A float which is added to the randomly generated powers of teams in conference A but not B
    comp: bool optional
        If true, measured conference differences are compared to true conference differences, if not measured 
        conference differences are compared to expected conference differences    
        
            
    Returns
    ---------
    d: numpy array
        List of measured conference power differences
    '''
    if comp:
        d = np.zeros(reps)
        for i in range(reps):
            result = simulate(n,m,M,g,dist,bias)
            W=result[0]
            pitrue = result[1] 
            pi = RankZermelo(W,powers=True)
            d[i]=scoreDistanceComp(pi,pitrue,n,m,t=t)
    else:
        d = np.zeros(reps)
        for i in range(reps):
            result = simulate(n,m,M,g,dist,bias)
            W=result[0]
            pi = RankZermelo(W,powers=True)
            d[i]=scoreDistance(pi,n,m,t=t)
    return(d)

def difRankDistribution(n,m,M,t=None,reps=2000, g=9,dist = 'uniform', bias=0,comp = False):
    '''
    Finds an emperical distribution of observed differences between average conference ranks

    Parameters
    ----------
    n: int
        size of confernce A
    m: int
        size of conference B
    M: int
        Total number of interconference games
    t: int optional
        The number of teams measured from the top
    reps: int optional
        Number of simulated seasons measured
    g: int optional
        Number of conference games 
    dist: string optional
        If "uniform" the true powers are generated uniformly. If not they are generated exponentially.
    bias: float, optional
        A float which is added to the randomly generated powers of teams in conference A but not B
    comp: bool optional
        If true, measured conference differences are compared to true conference differences, if not measured 
        conference differences are compared to expected conference differences    
        
            
    Returns
    ---------
    d: numpy array
        List of measured conference rank differences
    '''
    if comp:
        d = np.zeros(reps)
        for i in range(reps):
            result = simulate(n,m,M,g,dist,bias)
            W=result[0]
            pitrue = np.argsort(result[1]) 
            pi = RankZermelo(W,powers=False)
            d[i]=rankDistanceComp(pi,pitrue,n,m,t)
    else:
        d = np.zeros(reps)
        for i in range(reps):
            result = simulate(n,m,M,g,dist,bias)
            W=result[0]
            pi = RankZermelo(W,powers=False)
            d[i]=rankDistance(pi,n,m,t)
    return(d)

def get_ecdf(data):
    '''
    Generates a cumulative distribution function from an emperical distribution

    Parameters
    ----------
    data: numpy array
        1D numpy array with float data
            
    Returns
    ---------
    data_sorted: numpy array
        A list of data points sorted in ascending order
    p: numpy array
        The proportion of data points which preceed the current data point.

    '''
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return([data_sorted,p])

def numericalExp2(n,m,bmin, bmax,bres = 100, Ms = [0,3,6,9],reps = 100,g=9,dist = "exp", info = 1, spread = 0, modular_test = False):
    '''
    Observes how bias persists in the Elo ranking method over a single football season with different numbers of interconference games

    Parameters
    ----------
    n: int
        number of teams in conference A
    m: int
        number of teams in conference B
    bmin: float
        minimum ELO bias measured
    bmax: float
        maximum ELO bias measured
    bres: int optional
        number of different biases measured in the interval [bmin, bmax]
    Ms: numpy array optional
        array of ints describing many interconference games to include in each trial
    reps: int optional
        number of repititions for each bias, ICG combination
    g : int optional
        number of conference games played by each team
    dist: string optional
        If "uniform" team powers are generated uniformly, otherwise they are generated exponentially
    info: float optional
        the fidelity of the initial ELO guess to the true expected ELO
    spread: float optional
        The standard deviation in the noise added to the initial ELO guess
    modular_test: bool optional
        If true, experiment is run with no interconference games, just additional conference games
            
    Returns
    ---------
    df: pandas DataFrame
        A dataframe of the experimental results
    '''
    df = pd.DataFrame(columns=["M","b","dif","info","spread"])
    for b in np.linspace(bmin,bmax,bres):
        print("{}% Complete".format(b/bmax*100))
        for M in Ms:
            for k in range(0,reps):
                result = simulateEnforced(n,m,M, g=g,dist=dist,comparison=modular_test)
                W=result[0]
                pi = result[1]
                initialELO = (info*(np.log(pi)*100+1500))+ b*np.concatenate((np.ones(n),np.zeros(m)))
                finalELO = RankELO(W,initialELO,returnRat = True)
                meanA = np.mean(finalELO[0:n])
                meanB = np.mean(finalELO[n:])
                dif = meanA-meanB
                newrow = pd.DataFrame({"M":M,"b":b,"dif":dif, "info":info, "spread":spread},index = [0])
                df=pd.concat([df,newrow])
    return(df)
