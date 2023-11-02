import csv
import os

os.makedirs('./data', exist_ok=True)
with open("ame-data-oct5.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i =0 
    for row in spamreader:
        open(f"data/{row[0]}.txt" , "w").write(row[8])
