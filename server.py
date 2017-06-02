from flask import Flask, jsonify, request, abort


"""DEPENDENCIES"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import linregress
import scipy.integrate as integrate
from sklearn import linear_model
import json
import math
import pprint



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

with open("challenger.json") as openfile:
    data = json.load(openfile)
challenger_players_data = data
challenger_players_stats = []
num_master_players = 42

def clean_and_extract_stats(data):
    x = []
    for i in range(42):
        for j in range(10):
            if data[i][j]["gameMode"] == "CLASSIC":
                x.append(data[i][j]["stats"])
    return x


challenger_players_stats = clean_and_extract_stats(challenger_players_data)

least_games = len(challenger_players_stats)

def record_wins(d):
    if d == False:
        return 0
    else:
        return 1


challenger_gold_earned = [challenger_players_stats[i]["goldEarned"]for i in range(least_games) if "goldEarned" in challenger_players_stats[i]]
challenger_gold_spent = [challenger_players_stats[i]["goldSpent"]for i in range(least_games) if "goldSpent" in challenger_players_stats[i]]
challenger_time = [challenger_players_stats[i]["timePlayed"]for i in range(least_games)]
challenger_dmg_taken = [challenger_players_stats[i]["totalDamageTaken"]for i in range(least_games)]
challenger_neutral_minions = [challenger_players_stats[i]["minionsKilled"] for i in range(least_games) if "minionsKilled" in challenger_players_stats[i]]
challenger_minions = [challenger_players_stats[i]["neutralMinionsKilled"] for i in range(least_games) if "neutralMinionsKilled" in challenger_players_stats[i]]
challenger_wins = [record_wins(challenger_players_stats[i]["win"]) for i in range(least_games)]
challenger_kills = [challenger_players_stats[i]["championsKilled"] for i in range(least_games) if "championsKilled" in challenger_players_stats[i]]
challenger_deaths = [challenger_players_stats[i]["numDeaths"] for i in range(least_games) if "numDeaths" in challenger_players_stats[i]]

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
    print p
    print x

    total = getProbability(request.json["start"], request.json["end"], 1, mu, std, d)
    prob = {
        "pdf": total,
        "norm": p.tolist(),
        "x": x.tolist(),
        "mean": mu,
        "std": std
    }
    return jsonify(prob), 201


@app.route('/ml/api/v1.0/data/linear-regression', methods=["POST"])
def linear_regression():
    if not request.json:
        abort(400)
    x = challenger_gold_earned
    y = challenger_gold_spent
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    data = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std": std_err
    }
    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/logistic-regression', methods=["POST"])
def logisitic_regression():
    if not request.json:
        abort(400)
    cge = challenger_gold_earned
    cgs = challenger_gold_spent
    ct = challenger_time
    wins = challenger_wins
    cd = challenger_deaths
    x = np.matrix([cge, cgs, ct])
    y = np.array(wins)
    print y
    x = x.transpose()
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(x, y)
    t = np.matrix([[1,1, 1 ], [1, 1, 1], [1, 2, 2]])
    t = t.transpose()
    print clf.predict(t)
    print clf.score(t, y)
    data = {

    }
    return jsonify(data), 201
