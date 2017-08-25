import csv
import json

# csvfile = open('csv-sales-2015.csv', 'r')
# jsonfile = open('csv-sales-2015.json', 'w')
# # "Balance","Amount","Sales Price","U/M","Qty","Item","Name","Memo","Num","Date","Type"
# fieldnames = ("Type","Date","U/M","Num","Name","Item","Qty","Memo","Sales Price","Amount","Balance")
# reader = csv.DictReader(csvfile, fieldnames)
# data = {"sales":[]}
# for row in reader:
#     json.dump(row, jsonfile)
#     jsonfile.write('\n')

# obj = {"months":
#         [
#             {
#                 "Jan": []
#             },
#             {
#                 "Feb": []
#             },
#             {
#                 "Mar": []
#             },
#             {
#                 "Apr": []
#             },
#             {
#                 "May": []
#             },
#             {
#                 "Jun": []
#             },
#             {
#                 "Jul": []
#             },
#             {
#                 "Aug": []
#             },
#             {
#                 "Sep": []
#             },
#             {
#                 "Oct": []
#             },
#             {
#                 "Nov": []
#             },
#             {
#                 "Dec": []
#             },
#
#
#         ]
# }
# counter = 0
# with open('csv-sales-2015.json') as json_data:
#     d = json.load(json_data)
#     for i in d["Data"]:
#         counter += 1
#         print counter
#         print i["Name"] + " " + i["Date"]
#         if "01/" in i["Date"]:
#             obj["months"][0]["Jan"].append(i);
#         if "02/" in i["Date"]:
#             obj["months"][1]["Feb"].append(i);
#         if "03/" in i["Date"]:
#             obj["months"][2]["Mar"].append(i);
#         if "04/" in i["Date"]:
#             obj["months"][3]["Apr"].append(i);
#         if "05/" in i["Date"]:
#             obj["months"][4]["May"].append(i);
#         if "06/" in i["Date"]:
#             obj["months"][5]["Jun"].append(i);
#         if "07/" in i["Date"]:
#             obj["months"][6]["Jul"].append(i);
#         if "08/" in i["Date"]:
#             obj["months"][7]["Aug"].append(i);
#         if "09/" in i["Date"]:
#             obj["months"][8]["Sep"].append(i);
#         if "10/" in i["Date"]:
#             obj["months"][9]["Oct"].append(i);
#         if "11/" in i["Date"]:
#             obj["months"][10]["Nov"].append(i);
#         if "12/" in i["Date"]:
#             obj["months"][11]["Dec"].append(i);
#
#
# print obj["months"][0]["Jan"]

# jsonfile = open('csv-sales-by-month-2015.json', 'w')
# json.dump(obj, jsonfile, indent=4)
# jsonfile.write('\n')

with open('csv-sales-by-month-2015.json') as json_data:
    d = json.load(json_data)
    jan = d["months"][0]["Jan"]
    feb = d["months"][1]["Feb"]
    mar = d["months"][2]["Mar"]
    apr = d["months"][3]["Apr"]
    may = d["months"][4]["May"]
    jun = d["months"][5]["Jun"]
    jul = d["months"][6]["Jul"]
    aug = d["months"][7]["Aug"]

    jan_sales = 0
    for i in jan:
        i["Amount"].replace("'","").replace("\n","")
        jan_sales += float(i["Amount"])
    print jan_sales

    feb_sales = 0
    for i in feb:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        feb_sales += float(i["Amount"])
    print feb_sales

    mar_sales = 0
    for i in mar:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        mar_sales += float(i["Amount"])
    print mar_sales

    apr_sales = 0
    for i in apr:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        apr_sales += float(i["Amount"])
    print apr_sales

    may_sales = 0
    for i in may:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        may_sales += float(i["Amount"])
    print may_sales

    jun_sales = 0
    for i in jun:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        jun_sales += float(i["Amount"])
    print jun_sales

    jul_sales = 0
    for i in jul:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        jul_sales += float(i["Amount"])
    print jul_sales

    aug_sales = 0
    for i in aug:
        # i["Amount"].replace("'","").replace("\n","")
        # print i["Amount"]
        aug_sales += float(i["Amount"])
    print aug_sales
