"""This script loads the metadata for the maath papers from the Arxiv for the last five years.
It will then use the Arxiv API to harvest the MSC whenever they are available. Note that
the terms of use of the Arxiv API.
https://arxiv.org/help/api/index
THIS SCRIPT IS NOT USEFUL. IT IS NOT POSSIBLE TO HARVEST THE MSC FROM THE ARXIV WEBSITE. LEFT FOR LEGACY.
"""
import os
import urllib, urllib.request
from urllib.error import HTTPError
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Load the metadata

PATH_TO_METADATA_FOLDER = "data/metadata/"
PATH_TO_METADATA = PATH_TO_METADATA_FOLDER + "df_maths.json"
PATH_TO_METADATA_MSC = PATH_TO_METADATA_FOLDER + "df_maths_msc.json"
DICT_TYPE = {"id": str}

# Check the MSC have been harvested yet.
if os.path.exists(PATH_TO_METADATA_MSC):
    df = pd.read_json(PATH_TO_METADATA_MSC, dtype=DICT_TYPE)
else:
    df = pd.read_json(PATH_TO_METADATA, dtype=DICT_TYPE)

# Check the read ids./
mask_to_read = df["read"] == False
df_to_read = df.loc[mask_to_read, :]
counter = 0
# Harvesting
try:
    for id in tqdm(df_to_read["id"]):
        counter += 1
        mask_id = df["id"] == id
        url = "http://export.arxiv.org/abs/" + f"{id}"
        try:
            if counter % 4 == 0:
                time.sleep(1)
            webdata = urllib.request.urlopen(url)
            page = webdata.read().decode("utf-8")
        except HTTPError:
            print("connection error, making a break")
            time.sleep(60)
            print("trying again")
        else:
            m = re.search("""<td class="tablecell msc-classes">(.+?)</td>""", page)
            if m != None:
                msc = m.group(1)
                df.loc[mask_id, ["read", "msc"]] = [True, msc]
            else:
                df.loc[mask_id, ["read"]] = [True]    
finally:
    df.to_json(PATH_TO_METADATA_FOLDER + "df_maths_msc.json")
