import time

import pandas as pd
import copy
import re
from nltk.stem import PorterStemmer
import json
from nltk.tokenize import RegexpTokenizer
from itertools import chain

df = pd.read_json("full_format_recipes.json")  # Kaggle DataBase
instructions = df.directions.dropna()  # Parses out the recipe instructions
my_object = list(chain.from_iterable(instructions.tolist()))  # flats the data
as_one_str = ''.join(my_object)  # instructions: list to string

# split the string into sentences (i.e. according to a "." that comes after a word)
lst_of_sentences = re.split("((?<=[A-Za-z])+\.)", as_one_str)
# store the sentences in a Pandas Series
corpus = pd.Series(lst_of_sentences)
corpus = corpus[corpus != "."]
corpus = corpus.str.lower()
corpus = corpus.apply(lambda sentence: RegexpTokenizer(r'\w+').tokenize(sentence))
print("sentences split")


# function to apply each sentence in the corpus
# tokenized_sentence is a list
# techs is a series of stemmed cook-techniques
def stem_techs(tokenized_sentence, techs):
    pst = PorterStemmer()
    output_sentence = copy.deepcopy(tokenized_sentence)
    words_that_stemmed = []
    for i, word in enumerate(tokenized_sentence):
        stemmed = pst.stem(word)
        if (techs == stemmed).any():
            output_sentence[i] = stemmed
            words_that_stemmed.append(stemmed)
    return pd.Series([output_sentence, words_that_stemmed])


"""Cell to apply desired stemmings, approx 15 min"""
print("cook tech stem start")
# Create a new Series of STEMMED techniques
cook_techs = pd.read_csv("raw_techs.csv")  # cooking techniques DB
ps = PorterStemmer()
cook_techs["stemmed"] = cook_techs["tech"].apply(lambda x: ps.stem(x))

# raw_techs['stemmed'] is the stemmed series of raw_techs
df_after_stem = corpus.apply(lambda sentence: stem_techs(sentence, cook_techs['stemmed']))
print("cook tech stem end")
df_after_stem.columns = ['sentence', 'stemmed']
# join each sentence into a single string
df_after_stem['sentence'] = df_after_stem['sentence'].apply(lambda x: " ".join(x))
df_after_stem.to_csv(r'df_after_stem.csv', index=None, header=True)  # save stemmed corpus

# create cooking technique histogram
cook_tech_hist = pd.Series(list(chain.from_iterable(df_after_stem["stemmed"].tolist()))).value_counts()
cook_tech_hist.to_csv(r'cook_tech_hist.csv', index=None, header=True)  # save techniques histogram

# read Kaggles ingredients DB
with open("train_ingredients.json", "r") as read_file:
    data_train = json.load(read_file)
train_ingredients = pd.read_json("train_ingredients.json")

with open("test_ingredients.json", "r") as read_file:
    data_test = json.load(read_file)
test_ingredients = pd.read_json("test_ingredients.json")

# filter the ingredients from the DB
train_ingredients = train_ingredients["ingredients"]
test_ingredients = test_ingredients["ingredients"]
ingredients = train_ingredients.append(test_ingredients)
ingredients = pd.Series(list(chain.from_iterable(ingredients.tolist())))

# remove duplicates
ingredients = pd.Series(ingredients.unique())

# not consider 'dash' or 'space' delimiter
to_append1 = ingredients.str.replace(" ", "-")
to_append2 = ingredients.str.replace("-", " ")
ingredients = ingredients.append(to_append1).append(to_append2)

# create 'single' ingredients (potatoes-->potatoe)
to_append3 = pd.Series(ingredients.str.replace("s{1}$", ""))
to_append4 = pd.Series(ingredients.str.replace("es{1}$", ""))
ingredients = ingredients.append(to_append3).append(to_append4)
# remove duplicates
ingredients = pd.Series(ingredients.unique())

# keep only the instructions from the corpus
corpus = df_after_stem['sentence']
print(corpus.head())

try:
    print("ing for loop start")
    print(time.ctime())
    # 1) remove ing which do not appear in the corpus
    # 2) replace 'space' with 'dash' for
    for ing in ingredients:
        # remove if not in the corpus more-than-one-word ingredients
        if not corpus.str.contains(ing).any():
            ingredients = ingredients.drop(ingredients[ingredients == ing].index)

        # merge all more-than-one-word ingredients to be separated by a dash
        elif bool(re.match(".* .*", ing)):
            dashed = re.sub(" ", "-", ing)
            ingredients = ingredients.str.replace(ing, dashed)
            corpus = corpus.str.replace(ing, dashed)

    print("ing for loop end")

    ingredients.to_csv("ingredients_of_corpus.csv")
    corpus.to_csv("with_dash_corpus.csv")

except:
    print("failed at")
    print(time.ctime())
else:
    print("great success")
    print(time.ctime())
