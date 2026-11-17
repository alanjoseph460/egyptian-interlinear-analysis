import csv, sys, random
sys.stdout.reconfigure(encoding="utf-8")
rows = list(csv.DictReader(open("solutions/working/egyptian-interlinear-analysis/dataset/public/train.csv", encoding="utf-8")))
random.seed(1)
for r in random.sample(rows, 14):
    print(r["latest_century"], r["n_words"])
    for h, a in zip(r["hieroglyphs"].split(" "), r["prediction"].split(" ")):
        print("   ", h, " ", a)
