import os
import re
#os.getcwd()
#os.chdir('C:/Users/Caio/repos/nba-models')

dir_str = "fasttext/aclImdb/train/neg/"
dir_str = "fasttext/aclImdb/train/pos/"
#directory = os.fsencode(dir_str)

out_str = "fasttext/imdb.txt"
out = open(out_str, "w", encoding="utf8")

for file in os.listdir(dir_str):
    ## get filename
    filename = os.fsdecode(file)
    f = open(dir_str + file,"r",encoding="utf8")

    ## format and write into output
    if filename.endswith(".txt"):
        review = f.read()
        review = strip_formatting(review)
        score = filename.split("_")[1].split(".")[0]
        print("__label__" + score + " " + review, file=out)


def strip_formatting(string):
    string = string.lower()
    string = string.strip("\\n")
    #string = string.strip("<br/>")
    string = re.sub(r"([.!?,'/()<>])", r" \1 ", string)
    return string
