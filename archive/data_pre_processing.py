from nltk.stem import PorterStemmer
import pandas as pd

raw_techs = pd.read_csv("raw_techs.csv")  # cooking techniques DB

# Create a new Series of STEMMED techniques
ps = PorterStemmer()
raw_techs["stemmed"] = raw_techs["tech"].apply(lambda x: ps.stem(x))
export_csv = raw_techs.to_csv(r'raw_techs.csv', index = None, header=True)

# STEM all the techniques in the recipe instructions DB
