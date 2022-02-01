""" This script downloads arxiv metadata, and extract from them the necessary metadata.
The arxiv metadata are available
https://www.kaggle.com/Cornell-University/arxiv?select=arxiv-metadata-oai-snapshot.json
It is possible to harvest arxiv metadata, see here
https://arxiv.org/help/bulk_data

I assume here that the metadata has been downloaded from kaggle in a folder 
PATH_TO_METADATA = "data/metadata/"
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# Avoiding the copy to slice error
pd.options.mode.chained_assignment = None

PATH_TO_FOLDER = "data/metadata/"
PATH_TO_METADATA = "data/metatdata/arxiv-metadata-oai-snapshot.json"

# Date from which the data will be considered.

YEAR = 2010


def cleaning_df(df: pd.DataFrame, year: int = 16) -> pd.DataFrame:
    """ Perform the cleaning of the dataframe of metadata.
    Args: A dataframe provided by the Arxiv project as metadata. 
    Returns: A dataframe with columns dropped ["submitter", "comments", "journal-ref", "doi",\
     "report-no", "license", "versions", "authors"] and date format in the datetime format.
    Filters only math papers. Select papers from 2016 only.
    """
    DROPPED_COLUMNS = [
        "submitter",
        "comments",
        "journal-ref",
        "doi",
        "report-no",
        "license",
        "versions",
        "authors",
        "update_date",
    ]
    df_copy = df.drop(DROPPED_COLUMNS, axis=1, inplace=False)
    df_copy["int_date"] = [
        "".join(filter(lambda i: i.isdigit(), id))[:4] for id in df_copy["id"]
    ]
    df_copy["date"] = pd.to_datetime(df_copy["int_date"], format="%y%m")
    df_copy.drop("int_date", axis=1, inplace=True)
    mask_math = []
    for category in df_copy.loc[:, "categories"]:
        boolean = "math" in category
        mask_math.append(boolean)
    mask_date = df_copy["date"].dt.year >= year
    mask = mask_date & mask_math
    df_maths = df_copy.loc[mask, :]
    transformed_categories = [string.split(" ") for string in df_maths["categories"]]
    math_categories = []
    for list_categories in transformed_categories:
        list_math_categories = [
            string for string in list_categories if "math" in string
        ]
        math_categories.append(list_math_categories)
    main_math_categories = [list_string[0] for list_string in math_categories]
    df_maths["categories"] = transformed_categories
    df_maths["math_categories"] = math_categories
    df_maths["main_math_categories"] = main_math_categories
    return df_maths


CHUNKSIZE = 10000
DICT_TYPE = {"id": str}
df = pd.read_json(PATH_TO_METADATA, dtype=DICT_TYPE, lines=True, chunksize=CHUNKSIZE)

df_maths = pd.DataFrame()

for chunk_df in tqdm(df):
    df_int = cleaning_df(chunk_df, YEAR)
    df_maths = pd.concat([df_maths, df_int])


# Preparing the harvesting of the informations.
df_maths["read"] = False

# Preparing the metadata columns.
df_maths["msc"] = np.nan


# Adding one-hot encoded categories.
list_math_categories = []
list_other_categories = []
list_categories = df_maths["categories"]
for categories in list_categories:
    for string in categories:
        if "math" in string:
            list_math_categories.append(string)
        else:
            list_other_categories.append(string)


# Creating the list of all labels.
list_math_categories = list(set(list_math_categories))
list_other_categories = list(set(list_other_categories))
list_all = list_math_categories + list_other_categories


# One hot encoding the categories
df_int = pd.DataFrame(columns=list_all)
for category in list_all:
    df_int[f"{category}"] = (
        df_maths["categories"].apply(lambda x: category in x).astype(int)
    )


df_maths = pd.concat([df_maths, df_int], axis=1)

df_maths.to_json(PATH_TO_FOLDER + "df_maths.json")

df_maths.to_csv(PATH_TO_FOLDER + "df_maths.csv", sep=",")

print(df_maths.shape)

print("Cleaning done.")
