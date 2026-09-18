import pandas as pd
import numpy as np
import functions as fn
import PairwiseComparison as pc

np.set_printoptions(edgeitems=10, infstr='inf',
linewidth=300, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)


Data_2014 = pd.read_csv('2014_game_results.csv')
Data_2015 = pd.read_csv('2015_game_results.csv')
Data_2016 = pd.read_csv('2016_game_results.csv')
Data_2017 = pd.read_csv('2017_game_results.csv')
Data_2018 = pd.read_csv('2018_game_results.csv')
Data_2019 = pd.read_csv('2019_game_results.csv')
Data_2021 = pd.read_csv('2021_game_results.csv')
Data_2022 = pd.read_csv('2022_game_results.csv')
Data_2023 = pd.read_csv('2023_game_results.csv')
Data_2024 = pd.read_csv('2024_game_results.csv')
Data_2025 = pd.read_csv('2025_game_results.csv')

Years = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
Data_Collection = [Data_2014, Data_2015, Data_2016, Data_2017, Data_2018, Data_2019, Data_2021, Data_2022, Data_2023, Data_2024, Data_2025]
season_length = [14,13,14,14,14,15,14,14,14,15,15]


# Set up/Merge Datasets
cleaned_seasons = []
for year, dataset, l in zip(Years, Data_Collection, season_length):
    cleaned = fn.clean_season(dataset, year)
    cleaned = cleaned[cleaned['Wk']<=l]
    cleaned_seasons.append(cleaned)

all_games = pd.concat(cleaned_seasons, ignore_index=True)
all_games = all_games.sort_values('Date', kind='stable').reset_index(drop=True)

SEC1=["Alabama","Arkansas","Auburn","Florida","Georgia","Kentucky","Louisiana State","Mississippi State","Mississippi","Missouri","South Carolina","Tennessee","Texas A&M","Vanderbilt"]
SEC2=["Alabama","Arkansas","Auburn","Florida","Georgia","Kentucky","Louisiana State","Mississippi State","Mississippi","Missouri","Oklahoma", "South Carolina","Tennessee","Texas","Texas A&M","Vanderbilt"]
BIG1=["Illinois","Indiana","Iowa","Maryland","Michigan","Michigan State","Minnesota", "Nebraska","Northwestern","Ohio State","Penn State","Purdue","Rutgers","Wisconsin"]
BIG2=["Illinois","Indiana","Iowa","Maryland","Michigan","Michigan State","Minnesota", "Nebraska","Northwestern","Ohio State","Oregon","Penn State","Purdue","Rutgers","Southern California","UCLA","Washington","Wisconsin"]

Teams1 = SEC1+BIG1
Teams2 = SEC2+BIG2
reduced_seasons=[]
for year, dataset in zip(Years, cleaned_seasons):
    if year<=2023:
        cleaned = dataset[dataset.Winner.isin(Teams1)]
        cleaned = cleaned[cleaned.Loser.isin(Teams1)]
    else:
        cleaned = dataset[dataset.Winner.isin(Teams2)]
        cleaned = cleaned[cleaned.Loser.isin(Teams2)]

    reduced_seasons.append(cleaned)

all_games = pd.concat(reduced_seasons, ignore_index=True)
all_games = all_games.sort_values('Date', kind='stable').reset_index(drop=True)


score_matrices=[]
for year, dataset in zip(Years, reduced_seasons):
    if year <=2023:
        Teams = Teams1
    else:
        Teams=Teams2
    W=np.zeros((len(Teams),len(Teams)))
    for i in range(len(Teams)):
        teamdf = dataset[dataset.Winner == Teams[i]]
        for j in range(len(Teams)):
            game = teamdf[teamdf.Loser == Teams[j]]
            if not game.empty:
                W[i,j]=sum(game.PtsW)
                W[j,i]=sum(game.PtsL)
    score_matrices.append(W)



results = pd.DataFrame(columns=("year","ICG","difp","difr","dif3p","dif3r"))
for year, scores in zip(Years,score_matrices):
    if year <=2023:
        Teams = Teams1
        n=14
        m=14
    else:
        Teams=Teams2
        n=16
        m=18
    icgScores1 = scores[n:,:n]
    icgScores2 = scores[:n,n:]
    ICG = max(np.count_nonzero(icgScores1),np.count_nonzero(icgScores1))
    p = pc.RankZermelo(scores,powers=True)
    r = np.argsort(np.flip(np.argsort(p)))
    difr=pc.rankDistance(r,n,m)
    difp=pc.scoreDistance(p,n,m)
    dif3r=pc.rankDistance(r,n,m,3)
    dif3p=pc.scoreDistance(p,n,m,3)
    res = pd.DataFrame({"year":year,"ICG":ICG,"difp":difp,"difr":difr,"dif3p":dif3p,"dif3r":dif3r}, index=[0])
    results = np.concatenate((results,res))
print(results)

#Test if SEC is on average Better than the Big 10 by power
d1=pc.difRankDistribution(n,m,i,t=-1,reps=5000,bias = 0,comp=True)
d2=pc.difRankDistribution(n,m,i,t=-1,reps=5000,bias = 0,comp=True)
d3=pc.difRankDistribution(n,m,i,t=-1,reps=5000,bias = 0,comp=True)
d4=pc.difRankDistribution(n,m,i,t=-1,reps=5000,bias = 0,comp=True)
dlist = [d1,d2,d3,d4]

for i in range(len(results.year)):
    ICG = results.ICG[i]
    difp = results.difp[i]
    if year <=2023:
        Teams = Teams1
        n=14
        m=14
    else:
        Teams=Teams2
        n=16
        m=18
    d=dlist[ICG-1]
    [x,y]=pc.get_ecdf(d)
    idx =np.argmax(x>-difp)
    p=y[idx]
    print(p)


