from flask import Flask, jsonify, request, abort


"""DEPENDENCIES"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import linregress
import scipy.integrate as integrate
from sklearn import linear_model
import json
import math
import pprint
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, World!"

tasks = [
    {
        "id": 1,
        "data": [1, 2, 3, 4, 5, 6]
    }
]

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


challenger_players_stats = clean_and_extract_stats(challenger_players_data)
master_players_stats = clean_and_extract_stats(master_players_data)
bronze_players_stats = clean_and_extract_stats(bronze_players_data)

least_games = len(challenger_players_stats)
bronze_least_games = len(bronze_players_stats)

challenger_gold_earned = [challenger_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in challenger_players_stats[i]]
challenger_gold_spent = [challenger_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in challenger_players_stats[i]]
challenger_time = [challenger_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
challenger_dmg_taken = [challenger_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
challenger_neutral_minions = [challenger_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in challenger_players_stats[i]]
challenger_minions = [challenger_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in challenger_players_stats[i]]
challenger_wins = [record_wins(challenger_players_stats[i]["win"]) for i in range(bronze_least_games)]
challenger_kills = [challenger_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in challenger_players_stats[i]]
challenger_deaths = [challenger_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in challenger_players_stats[i]]

bronze_gold_earned = [bronze_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in bronze_players_stats[i]]
bronze_gold_spent = [bronze_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in bronze_players_stats[i]]
bronze_time = [bronze_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
bronze_dmg_taken = [bronze_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
bronze_neutral_minions = [bronze_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in bronze_players_stats[i]]
bronze_minions = [bronze_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in bronze_players_stats[i]]
bronze_wins = [record_wins(bronze_players_stats[i]["win"]) for i in range(bronze_least_games)]
bronze_kills = [bronze_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in bronze_players_stats[i]]
bronze_deaths = [bronze_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in bronze_players_stats[i]]

master_gold_earned = [master_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in master_players_stats[i]]
master_gold_spent = [master_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in master_players_stats[i]]
master_time = [master_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
master_dmg_taken = [master_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
master_neutral_minions = [master_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in master_players_stats[i]]
master_minions = [master_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in master_players_stats[i]]
master_wins = [record_wins(master_players_stats[i]["win"]) for i in range(bronze_least_games)]
master_kills = [master_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in master_players_stats[i]]
master_deaths = [master_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in master_players_stats[i]]



def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def getProbability(x, y, dx, mu, std, data):
    total = 0
    while x < y:
        total += (normpdf(x, mu, std) * dx)
        x += dx
    return total


@app.route('/ml/api/v1.0/data/pdf', methods=["POST"])
def get_probability():
    if not request.json:
        abort(400)

    d = challenger_gold_earned

    mu, std = norm.fit(d)

    plt.hist(d, bins=25, normed=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    total = getProbability(request.json["start"], request.json["end"], 1, mu, std, d)
    prob = {
        "pdf": total,
        "norm": p.tolist(),
        "x": x.tolist(),
        "mean": mu,
        "std": std
    }

    d = bronze_gold_earned

    mu, std = norm.fit(d)

    plt.hist(d, bins=25, normed=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    prob2 = {

        "pdf": total,
        "norm": p.tolist(),
        "x": x.tolist(),
        "mean": mu,
        "std": std
    }

    d = master_gold_earned

    mu, std = norm.fit(d)

    plt.hist(d, bins=25, normed=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    prob3 = {

        "pdf": total,
        "norm": p.tolist(),
        "x": x.tolist(),
        "mean": mu,
        "std": std
    }

    d = challenger_kills

    mu, std = norm.fit(d)

    plt.hist(d, bins=25, normed=True, alpha=0.6)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    total = getProbability(request.json["start"], request.json["end"], 1, mu, std, d)
    chall_kills = {
        "pdf": total,
        "norm": p.tolist(),
        "x": x.tolist(),
        "mean": mu,
        "std": std,
        "d": d
    }

    return jsonify([prob, prob2, prob3, chall_kills]), 201


@app.route('/ml/api/v1.0/data/linear-regression', methods=["POST"])
def linear_regression():
    if not request.json:
        abort(400)
    x = request.json["kills"]
    y = request.json["deaths"]
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    data = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std": std_err,
        "max_x": max_x,
        "max_y": max_y,
        "min_x": min_x,
        "min_y": min_y
    }

    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/logistic-regression', methods=["POST"])
def logisitic_regression():
    if not request.json:
        abort(400)

    # cge = challenger_gold_earned
    # cgs = challenger_gold_spent
    cgt = challenger_time
    ck = challenger_kills
    cm = challenger_minions
    wins = challenger_wins
    cd = challenger_deaths
    x = np.matrix([ck, cm, cgt])
    y = np.array(wins)
    x = x.transpose()
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(x, y)
    player_data = np.matrix([request.json["x"], request.json["y"], request.json["z"]])
    player_data = player_data.transpose()
    data = {
        "data": list(clf.predict(player_data))
    }
    return jsonify(data), 201


app.run(host="0.0.0.0 ", port="8000")
