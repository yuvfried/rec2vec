from nltk.stem import PorterStemmer
import pandas as pd

raw_techs = pd.read_csv("raw_techs.csv")  # cooking techniques DB
rec_inst = pd.DataFrame([["baking", "dry"],["frying", "wet"]])  # recipe instructions BD

# Create a new Series of STEMMED techniques
ps = PorterStemmer()
raw_techs["stemmed"] = raw_techs["tech"].apply(lambda x: ps.stem(x))

# STEM all the techniques in the recipe instructions DB
