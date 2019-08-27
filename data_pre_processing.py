from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd


raw_techs = pd.read_csv("raw_techs.csv")
# raw_techs = pd.DataFrame([["baking", "dry"],["frying", "wet"]])
print(raw_techs.to_string())

ps = PorterStemmer()
raw_techs = raw_techs.apply(lambda x: ps.stem(x) if x)
print(raw_techs)
# print(raw_techs[["tech", "count"]])

# words = ["bak", "bake", "baking", "bakes"]
# for word in words:
#     print(ps.stem(word))
