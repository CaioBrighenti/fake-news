import os
from os import path
from pprint import pprint
import platform
from pycorenlp.corenlp import StanfordCoreNLP
import nltk
from nltk.tree import *
import numpy as np
import pandas as pd

def getConstTreeDepths(text):
    host = "http://localhost"
    port = "9000"
    nlp = StanfordCoreNLP(host + ":" + port)
    output = nlp.annotate(
        text,
        properties={
            "outputFormat": "json",
            "annotators": "parse"
        }
    )
    sen_depth_list, NP_depth_list, VP_depth_list = list(), list(), list()
    ## check for timeout
    if type(output) == str:
        return {
        "sentence" : -1,
        "verb-phrase" : -1,
        "noun-phrase" : -1
        }
    ## if not timeout, loop through sentences
    for sen in output["sentences"]:
        parse_tree = nltk.Tree.fromstring(sen["parse"])
        sen_depth_list.append(parse_tree.height())
        ## find mean NP and VP depth
        for pos in parse_tree.treepositions():
            subtree = parse_tree[pos]
            ## leaf node
            if type(subtree) == str:
                continue
            label = subtree.label()
            if label == "NP":
                NP_depth_list.append(subtree.height())
            elif label == "VP":
                VP_depth_list.append(subtree.height())
    # avoid empty lists
    if len(sen_depth_list) == 0:
        sen_depth_list.append(0)
    if len(NP_depth_list) == 0:
        NP_depth_list.append(0)
    if len(VP_depth_list) == 0:
        VP_depth_list.append(0)
    # calculate medians
    depth_dict = {
    "mu-sentence" : np.mean(sen_depth_list),
    "mu-verb-phrase" : np.mean(VP_depth_list),
    "mu-noun-phrase" : np.mean(NP_depth_list),
    "sd-sentence" : np.std(sen_depth_list),
    "sd-verb-phrase" : np.std(VP_depth_list),
    "sd-noun-phrase" : np.std(NP_depth_list),
    "iqr-sentence" : np.subtract(*np.percentile(sen_depth_list, [75, 25])),
    "iqr-verb-phrase" : np.subtract(*np.percentile(VP_depth_list, [75, 25])),
    "iqr-noun-phrase" : np.subtract(*np.percentile(NP_depth_list, [75, 25]))
    }
    return depth_dict
