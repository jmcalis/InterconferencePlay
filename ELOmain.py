import pandas as pd
import functions as fn
from math import log

# Elo parameters
k = log(10)/400  # Logistic function steepness parameter
update_k = 20  # Elo update factor



# Import Datasets
Data_2014 = pd.read_csv('../Datasets/2014/2014_game_results.csv')
Data_2015 = pd.read_csv('../Datasets/2015/2015_game_results.csv')
Data_2016 = pd.read_csv('../Datasets/2016/2016_game_results.csv')
Data_2017 = pd.read_csv('../Datasets/2017/2017_game_results.csv')
Data_2018 = pd.read_csv('../Datasets/2018/2018_game_results.csv')
Data_2019 = pd.read_csv('../Datasets/2019/2019_game_results.csv')
Data_2021 = pd.read_csv('../Datasets/2021/2021_game_results.csv')
Data_2022 = pd.read_csv('../Datasets/2022/2022_game_results.csv')
Data_2023 = pd.read_csv('../Datasets/2023/2023_game_results.csv')
Data_2024 = pd.read_csv('../Datasets/2024/2024_game_results.csv')
Data_2025 = pd.read_csv('../Datasets/2025/2025_game_results.csv')

Years = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
Data_Collection = [Data_2014, Data_2015, Data_2016, Data_2017, Data_2018, Data_2019, Data_2021, Data_2022, Data_2023, Data_2024, Data_2025]


# Set up/Merge Datasets
cleaned_seasons = []
for year, dataset in zip(Years, Data_Collection):
    cleaned = fn.clean_season(dataset, year)
    cleaned_seasons.append(cleaned)
    print(year, 'missing winner scores:', cleaned['PtsW'].isna().sum(),
        'missing loser scores:', cleaned['PtsL'].isna().sum())

all_games = pd.concat(cleaned_seasons, ignore_index=True)
all_games = all_games.sort_values('Date', kind='stable').reset_index(drop=True)


initial_elo = 1500
ratings = {}


winner_probabilities = []
winner_pregame_elos = []
loser_pregame_elos = []


for index, row in all_games.iterrows():
    winner = row['Winner']
    loser = row['Loser']
    winner_elo = ratings.get(winner, initial_elo)
    loser_elo = ratings.get(loser, initial_elo)

    # Home-field advantage adjustment
    if row['Location'] == 'Home':
        h = 55  # Home-field advantage for the winner
    elif row['Location'] == 'Away':
        h = -55  # Home-field disadvantage for the winner
    else:
        h = 0  # Neutral field, no advantage

    winner_probability = fn.prob_win(
    winner_elo, loser_elo, k, h)

    winner_probabilities.append(winner_probability)
    winner_pregame_elos.append(winner_elo)
    loser_pregame_elos.append(loser_elo)

    # Update Elo ratings based on the game outcome
    updated_winner_elo, updated_loser_elo = fn.update_elo(winner_elo, loser_elo, k= k, K=update_k, outcome=1, h=h)

    # Store the updated Elo ratings back in the dictionary
    ratings[winner] = updated_winner_elo
    ratings[loser] = updated_loser_elo

all_games['WinnerEloPre'] = winner_pregame_elos
all_games['LoserEloPre'] = loser_pregame_elos
all_games['WinnerProbability'] = winner_probabilities

final_ratings = (
    pd.Series(ratings, name='Elo')
    .sort_values(ascending=False)
)

print(final_ratings.head(25))
print('Teams:', len(final_ratings))
print('Mean Elo:', final_ratings.mean())

accuracy = (all_games['WinnerProbability'] >= 0.5).mean()
print('Accuracy:', accuracy)



