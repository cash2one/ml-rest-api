from __future__ import division
import logging
from flask import Flask, jsonify, request, abort


"""DEPENDENCIES"""
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import linregress
import scipy.integrate as integrate
from scipy.integrate import quad
from sklearn import linear_model, svm
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
challenger_time = [challenger_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
challenger_dmg_taken = [challenger_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
challenger_neutral_minions = [challenger_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in challenger_players_stats[i]]
challenger_minions = [challenger_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in challenger_players_stats[i]]
challenger_wins = [record_wins(challenger_players_stats[i]["win"]) for i in range(bronze_least_games)]
challenger_kills = [int(challenger_players_stats[i]["championsKilled"]) for i in range(bronze_least_games) if "championsKilled" in challenger_players_stats[i]]
challenger_deaths = [challenger_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in challenger_players_stats[i]]

avg_challenger_gold_earned = sum(challenger_gold_earned) / bronze_least_games
avg_challenger_gold_spent = sum(challenger_gold_spent) / bronze_least_games
avg_challenger_time = sum(challenger_time) / bronze_least_games
avg_challenger_dmg_taken = sum(challenger_dmg_taken) / bronze_least_games
avg_challenger_minions = sum(minions_killed(challenger_neutral_minions, challenger_minions)) / bronze_least_games
avg_challenger_wins = sum(challenger_wins) / bronze_least_games
avg_challenger_kills = sum(challenger_kills) / bronze_least_games
avg_challenger_deaths  = sum(challenger_deaths) / bronze_least_games



bronze_gold_earned = [bronze_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in bronze_players_stats[i]]
bronze_gold_spent = [bronze_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in bronze_players_stats[i]]
bronze_time = [bronze_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
bronze_dmg_taken = [bronze_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
bronze_neutral_minions = [bronze_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in bronze_players_stats[i]]
bronze_minions = [bronze_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in bronze_players_stats[i]]
bronze_wins = [record_wins(bronze_players_stats[i]["win"]) for i in range(bronze_least_games)]
bronze_kills = [bronze_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in bronze_players_stats[i]]
bronze_deaths = [bronze_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in bronze_players_stats[i]]

avg_bronze_gold_earned = sum(bronze_gold_earned) / bronze_least_games
avg_bronze_gold_spent = sum(bronze_gold_spent) / bronze_least_games
avg_bronze_time = sum(bronze_time) / bronze_least_games
avg_bronze_dmg_taken = sum(bronze_dmg_taken) / bronze_least_games
avg_bronze_minions = sum(minions_killed(bronze_neutral_minions,bronze_minions)) / bronze_least_games
avg_bronze_wins = sum(bronze_wins) / bronze_least_games
avg_bronze_kills = sum(bronze_kills) / bronze_least_games
avg_bronze_deaths  = sum(bronze_deaths) / bronze_least_games

master_gold_earned = [master_players_stats[i]["goldEarned"]for i in range(bronze_least_games) if "goldEarned" in master_players_stats[i]]
master_gold_spent = [master_players_stats[i]["goldSpent"]for i in range(bronze_least_games) if "goldSpent" in master_players_stats[i]]
master_time = [master_players_stats[i]["timePlayed"]for i in range(bronze_least_games)]
master_dmg_taken = [master_players_stats[i]["totalDamageTaken"]for i in range(bronze_least_games)]
master_neutral_minions = [master_players_stats[i]["neutralMinionsKilled"] for i in range(bronze_least_games) if "neutralMinionsKilled" in master_players_stats[i]]
master_minions = [master_players_stats[i]["minionsKilled"] for i in range(bronze_least_games) if "minionsKilled" in master_players_stats[i]]
master_wins = [record_wins(master_players_stats[i]["win"]) for i in range(bronze_least_games)]
master_kills = [master_players_stats[i]["championsKilled"] for i in range(bronze_least_games) if "championsKilled" in master_players_stats[i]]
master_deaths = [master_players_stats[i]["numDeaths"] for i in range(bronze_least_games) if "numDeaths" in master_players_stats[i]]

avg_master_gold_earned = sum(master_gold_earned) / bronze_least_games
avg_master_gold_spent = sum(master_gold_spent) / bronze_least_games
avg_master_time = sum(master_time) / bronze_least_games
avg_master_dmg_taken = sum(master_dmg_taken) / bronze_least_games
avg_master_minions = sum(minions_killed(master_neutral_minions, master_minions)) / bronze_least_games
avg_master_wins = sum(master_wins) / bronze_least_games
avg_master_kills = sum(master_kills) / bronze_least_games
avg_master_deaths  = sum(master_deaths) / bronze_least_games


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
    y_final = (slope * max_x)+ intercept
    data = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std": std_err,
        "max_x": max_x,
        "max_y": max_y,
        "min_x": min_x,
        "min_y": min_y,
        "y_final": y_final
    }

    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/multiple-linear-regression', methods=["POST"])
def multiple_linear_regression():
    if not request.json:
        abort(400)

    x = request.json["x"]
    y = request.json["y"]
    z = request.json["z"]
    min_x = min(x)
    min_y = min(y)
    min_z = min(z)
    max_x = max(x)
    max_y = max(y)
    max_z = max(z)
    slope, intercept, r_value, p_value, std_err = linregress(x, y, z)
    y_final = (slope * max_x)+ intercept
    data = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std": std_err,
        "max_x": max_x,
        "max_y": max_y,
        "min_x": min_x,
        "min_y": min_y,
        "y_final": y_final
    }

    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/logistic-regression', methods=["POST"])
def logisitic_regression():

    if not request.json:
        abort(400)

    cge = challenger_gold_earned + master_gold_earned + bronze_gold_earned
    cgs = challenger_gold_spent + master_gold_spent + bronze_gold_spent
    cgt = challenger_time + master_time + bronze_time
    ck = challenger_kills + master_kills + bronze_kills
    wins = challenger_wins + master_wins + bronze_wins
    ck = np.matrix([challenger_kills + master_kills + bronze_kills])

    cd = challenger_deaths
    x = np.matrix([cge, cgs, cgt])
    y = np.array(wins)
    x = x.transpose()
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(x, y)
    player_data = np.matrix([request.json["x"], request.json["y"], request.json["z"]])
    player_data = player_data.transpose()
    proba = clf.predict_proba(x)
    score = clf.score(x, y)
    data = {
        "data": clf.predict(player_data).tolist(),
        "score": score,
    }
    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/central-tendencies', methods=["GET"])
def central_tendencies():

    if request.json:
        player_data = request.json["data"]
        kills = 0
        deaths = 0
        cs = 0
        assists = 0
        turret_kills = 0
        gold_earned = 0
        for i in range(len(player_data)):
            kills += player_data[i]["stats"]["kills"]
            deaths += player_data[i]["stats"]["deaths"]
            cs += player_data[i]["stats"]["totalMinionsKilled"]
            assists += player_data[i]["stats"]["assists"]
            turret_kills += player_data[i]["stats"]["turretKills"]
            gold_earned += player_data[i]["stats"]["goldEarned"]

        avg_kills = int(kills / len(player_data))
        avg_deaths = int(deaths / len(player_data))
        avg_cs = int(cs / len(player_data))
        avg_assists = int(assists / len(player_data))
        avg_turret_kills = turret_kills / len(player_data)
        avg_gold_earned = int(gold_earned / len(player_data))

        data = {
            "data": [avg_kills, avg_deaths, avg_cs, avg_assists, avg_turret_kills, avg_gold_earned],
            "master": [avg_master_gold_earned,
                        avg_master_gold_spent,
                        avg_master_time,
                        avg_master_dmg_taken,
                        avg_master_minions,
                        avg_master_wins, avg_master_kills,
                        avg_master_deaths],
            "bronze": [avg_bronze_gold_earned,
                        avg_bronze_gold_spent,
                        avg_bronze_time,
                        avg_bronze_dmg_taken,
                        avg_bronze_minions,
                        avg_bronze_wins, avg_master_kills,
                        avg_bronze_deaths],
            "challenger": [avg_challenger_gold_earned,
                        avg_challenger_gold_spent,
                        avg_challenger_time,
                        avg_challenger_dmg_taken,
                        avg_challenger_minions,
                        avg_challenger_wins, avg_master_kills,
                        avg_challenger_deaths],
        }
        return jsonify(data), 201
    data = {
        "master": [avg_master_gold_earned,
                    avg_master_gold_spent,
                    avg_master_time,
                    avg_master_dmg_taken,
                    avg_master_minions,
                    avg_master_wins, avg_master_kills,
                    avg_master_deaths],
        "bronze": [avg_bronze_gold_earned,
                    avg_bronze_gold_spent,
                    avg_bronze_time,
                    avg_bronze_dmg_taken,
                    avg_bronze_minions,
                    avg_bronze_wins, avg_master_kills,
                    avg_bronze_deaths],
        "challenger": [avg_challenger_gold_earned,
                    avg_challenger_gold_spent,
                    avg_challenger_time,
                    avg_challenger_dmg_taken,
                    avg_challenger_minions,
                    avg_challenger_wins, avg_master_kills,
                    avg_challenger_deaths],
    }
    return jsonify(data), 201


@app.route('/ml/api/v1.0/data/svm', methods=["POST"])
def support_vector_machine():

    if not request.json:
        abort(400)

    cge = challenger_gold_earned + master_gold_earned + bronze_gold_earned
    cgs = challenger_gold_spent + master_gold_spent + bronze_gold_spent
    cgt = challenger_time + master_time + bronze_time
    wins = challenger_wins + master_wins + bronze_wins
    print challenger_wins
    x = np.matrix([cge, cgs, cgt])
    y = np.array(wins)
    x = x.transpose()
    clf = svm.SVC(C=1.0, tol=1e-10, cache_size=600, kernel='linear',
              class_weight='auto')
    clf.fit(x, wins)
    player_data = np.matrix([request.json["x"], request.json["y"], request.json["z"]])
    player_data = player_data.transpose()
    data = {
        "data": list(clf.predict(player_data))
    }
    return jsonify(data), 201

@app.route('/ml/api/v1.0/data/integrate', methods=["POST"])
def integrate_():
    if not request.json:
        abort(400)

    points = request.json["data"]
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]

    # calculate polynomial
    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)
    print f
    def integrand(x, a, b, c):
        return (a*x**2) + (b * x) + c
    I = quad(integrand, 1, 5, args=(f[2], f[1], f[0]))
    data = {
        "data": I * 00.1
    }

    return jsonify(data), 201


app.run(host="0.0.0.0", port="8000")
