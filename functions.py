import numpy as np
import pandas as pd

def logit(x, k):
    """ Compute the logistic function of x using the exponential function. """
    return 1/(1 + np.exp(-k*x))

def prob_win(A, B, k, h=0):
    """ Compute the probability of player A beating player B"""
    return logit(A-B+h, k)

def update_elo(elo_A, elo_B, k,K, outcome, h=0):
    prob_A_wins = prob_win(elo_A, elo_B, k, h)

    if outcome == 1: # A wins
        elo_A_new = elo_A +  K*(1 - prob_A_wins)
        elo_B_new = elo_B - K*(1 - prob_A_wins)
    elif outcome == 0:
        elo_A_new = elo_A -  K*(prob_A_wins)
        elo_B_new = elo_B + K*(prob_A_wins)
    else:
        raise ValueError("Invalid outcome. Must be a 1 (A wins) or 0 (B wins).")
    return elo_A_new, elo_B_new

def clean_season(raw_dataframe, season):
    season_data = raw_dataframe.copy()
    season_data = season_data.rename({'Unnamed: 7': 'Location'}, axis = "columns")
    season_data['Location'] = (
        season_data['Location']
        .replace({'N': 'Neutral', '@': 'Away'})
        .fillna('Home')
    )

    season_data = season_data.drop(columns =["Rk"])
    season_data['Season'] = season
    season_data['Date'] = pd.to_datetime(season_data['Date'])

    if 'Pts' in season_data.columns and 'Pts.1' in season_data.columns:
        season_data = season_data.rename(
        columns={'Pts': 'PtsW', 'Pts.1': 'PtsL'}
    )

    for team_column, rank_column in [
        ('Winner', 'WinnerPollRank'),
    ('Loser', 'LoserPollRank')
    ]:
        season_data[rank_column] = (
        season_data[team_column]
        .str.extract(r'^\((\d+)\)', expand=False)
        .astype('Int64')
    )

        season_data[team_column] = (
        season_data[team_column]
        .str.replace(r'^\(\d+\)\s*', '', regex=True)
    )   

    season_data = season_data.dropna(subset=['PtsW', 'PtsL'])

    return season_data