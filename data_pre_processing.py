from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

ps = PorterStemmer()
raw_techs = pd.read_csv("raw_techs.csv")
rec_inst = pd.DataFrame([["baking", "dry"],["frying", "wet"]])

# Create a new Series of STEMMED techniques
raw_techs["stemmed"] = raw_techs["tech"].apply(lambda x: ps.stem(x))

# STEM all the techniques in the recipe instructions DB


# words = ["bak", "bake", "baking", "bakes"]
# for word in words:
#     print(ps.stem(word))
