import json
import pprint
import requests

from time import sleep

#200704989

q = requests.get("https://na1.api.riotgames.com/lol/match/v3/matchlists/by-account/200704989/recent?api_key=RGAPI-76173335-a3c4-4266-bb88-26284a77aa03")
matchlist_cleaned = q.json()["matches"]

pp = pprint.PrettyPrinter(indent=4)

games_list = []
for i in range(15):
    games_list.append(matchlist_cleaned[i]["gameId"])

cleaned_games_list = []
for i in range(len(games_list)):
    game = str(games_list[i])
    api_string = "https://na1.api.riotgames.com/lol/match/v3/matches/"+game+"?api_key=RGAPI-76173335-a3c4-4266-bb88-26284a77aa03"
    r = requests.get(api_string)
    cleaned = r.json()
    sleep(1)
    cleaned_games_list.append(cleaned)

game_identities = []
for i in range(len(cleaned_games_list)):
    cleaned = cleaned_games_list[i]
    for j in range(10):
        name = cleaned["participantIdentities"][j]["player"]["summonerName"]
        print name
        if (name == "NetseAl"):
            game_identities.append(cleaned["participantIdentities"][j]["participantId"])

timelines = []
# for i in range(len(game_identities)):
game = str(games_list[0])
f = requests.get("https://na1.api.riotgames.com/lol/match/v3/timelines/by-match/"+game+"?api_key=RGAPI-76173335-a3c4-4266-bb88-26284a77aa03")
f = f.json()
# pp.pprint(f)
# sleep(1)
timelines.append(f["frames"])

# for i in range(len(timelines)):
#     selection = timelines[0][i]
    # for j in range()["participantIdentities"]
    # pp.pprint(selection)
    

for i in range(len(timelines[0])):
    if timelines[0][i]["participantFrames"][unicode(game_identities[0])]:
        pp.pprint(timelines[0][i]["participantFrames"][unicode(game_identities[0])])
