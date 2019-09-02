import pandas as pd
import json
import re

# creates list of ingredients and replaces ' ' delimiter with "-" in more-than-on-word ingredients in the corpus

with open("/content/drive/My Drive/Final Project/train_ingredients.json", "r") as read_file:
    data = json.load(read_file)
train_ingredients = pd.read_json("/content/drive/My Drive/Final Project/train_ingredients.json")

with open("/content/drive/My Drive/Final Project/test_ingredients.json", "r") as read_file:
    data = json.load(read_file)
test_ingredients = pd.read_json("/content/drive/My Drive/Final Project/test_ingredients.json")

train_ingredients = train_ingredients["ingredients"]
test_ingredients = test_ingredients["ingredients"]
ingredients = train_ingredients.append(test_ingredients)
ingredients = pd.Series(list(chain.from_iterable(ingredients.tolist())))

# remove duplicates
ingredients = pd.Series(ingredients.unique())

# not consider 'dash' or 'space' delimiter
to_append1 = ingredients.str.replace("\s", "-")
to_append2 = ingredients.str.replace("-", "\s")
ingredients = ingredients.append(to_append1).append(to_append2)

# create 'single' ingredients (potatoes-->potato)
to_append3 = pd.Series(ingredients.str.replace("s{1}$", ""))
to_append4 = pd.Series(ingredients.str.replace("es{1}$", ""))
ingredients = ingredients.append(to_append3).append(to_append4)

# remove duplicated ingredients (in case that the last process created them)
ingredients = pd.Series(ingredients.unique())

# ingredient = intersect_of(ingredients, corpus_ingredients)
# remove more-than-one-word ingredients with ' ' delimiter (only '-' delimiter version will remain)
for ing in ingredients:
    # remove if not in the corpus
    if not corpus.str.contains(ing).any():
        ingredients = ingredients.drop(ingredients[ingredients == "eggs"].index)

    # unify all more-than-one-word ingredients to be separated by a dash
    elif bool(re.match(".* .*", ing)):
        dashed = re.sub(" ", "-", ing)
        ingredients = ingredients.str.replace(ing, dashed)
        corpus = corpus.str.replace(ing, dashed)

# write processed data to files
ingredients.to_csv("ingredients_of_corpus.csv")
corpus.to_csv("with_dash_corpus.csv")
