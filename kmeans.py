from sklearn.cluster import KMeans
import numpy as np

"""DEPENDENCIES"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import linregress
import scipy.integrate as integrate
from sklearn import linear_model, svm
import json
import math
import pprint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def clean_and_extract_stats(data):
    x = []
    for i in range(42):
        for j in range(10):
            if data[i][j]["gameMode"] == "CLASSIC":
                x.append(data[i][j]["stats"])
    return x

def record_wins(d):
    if d == False:
        return 0
    else:
        return 1

with open("challenger.json") as openfile:
    data = json.load(openfile)
challenger_players_data = data
challenger_players_stats = []
num_master_players = 42

with open("bronze.json") as openfile:
    data = json.load(openfile)
bronze_players_data = data
bronze_players_stats = []
num_master_players = 42

with open("master.json") as openfile:
    data = json.load(openfile)
master_players_data = data
master_players_stats = []
num_master_players = 42


def minions_killed(ls1, ls2):
    new_list = []
    for i in range(len(ls1)):
        new_list.append(ls1[i]+ls2[i])
    return new_list


challenger_players_stats = clean_and_extract_stats(challenger_players_data)
master_players_stats = clean_and_extract_stats(master_players_data)
bronze_players_stats = clean_and_extract_stats(bronze_players_data)

least_games = len(challenger_players_stats)
bronze_least_games = len(bronze_players_stats)

challenger_gold_earned = [challenger_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in challenger_players_stats[i]]
challenger_gold_spent = [challenger_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in challenger_players_stats[i]]
challenger_time = [challenger_players_stats[i]["timePlayed"]for i in range(20)]
challenger_dmg_taken = [challenger_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
challenger_neutral_minions = [challenger_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in challenger_players_stats[i]]
challenger_minions = [challenger_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in challenger_players_stats[i]]
challenger_wins = [record_wins(challenger_players_stats[i]["win"]) for i in range(bronze_least_games)]
challenger_kills = [int(challenger_players_stats[i]["championsKilled"]) for i in range(bronze_least_games) if "championsKilled" in challenger_players_stats[i]]
challenger_deaths = [challenger_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in challenger_players_stats[i]]


bronze_gold_earned = [bronze_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in bronze_players_stats[i]]
bronze_gold_spent = [bronze_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in bronze_players_stats[i]]
bronze_time = [bronze_players_stats[i]["timePlayed"]for i in range(20)]
bronze_dmg_taken = [bronze_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
bronze_neutral_minions = [bronze_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in bronze_players_stats[i]]
bronze_minions = [bronze_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in bronze_players_stats[i]]
bronze_wins = [record_wins(bronze_players_stats[i]["win"]) for i in range(bronze_least_games)]
bronze_kills = [bronze_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in bronze_players_stats[i]]
bronze_deaths = [bronze_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in bronze_players_stats[i]]

master_gold_earned = [master_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in master_players_stats[i]]
master_gold_spent = [master_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in master_players_stats[i]]
master_time = [master_players_stats[i]["timePlayed"]for i in range(20)]
master_dmg_taken = [master_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
master_neutral_minions = [master_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in master_players_stats[i]]
master_minions = [master_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in master_players_stats[i]]
master_wins = [record_wins(master_players_stats[i]["win"]) for i in range(bronze_least_games)]
master_kills = [master_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in master_players_stats[i]]
master_deaths = [master_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in master_players_stats[i]]


from sklearn.metrics import mean_squared_error, r2_score

X = np.array([challenger_time, master_time, bronze_time])
y = np.array([challenger_minions, master_minions, bronze_minions])
ols = linear_model.LinearRegression()
model = ols.fit(X, y)
print model.predict(challenger_time)
print model.intercept_

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(challenger_time, bronze_time, master_time, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
# c = c.reshape(-1, 1)
# print c.shape
# y = np.array([bronze_minions])
# print y.shape
# print np.dot(c, y)
# # print model.score(c, y)
plt.show()