"""This script loads the metadata for the maath papers from the Arxiv for the last five years.
It will then use the Arxiv API to harvest the MSC whenever they are available. Note that
the terms of use of the Arxiv API.
https://arxiv.org/help/api/index
THIS SCRIPT IS NOT USEFUL. IT IS NOT POSSIBLE TO HARVEST THE MSC FROM THE ARXIV WEBSITE. LEFT FOR LEGACY.
"""
import os
import urllib, urllib.request
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

# Load the metadata

PATH_TO_METADATA_FOLDER = "data/metatdata/"
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

# Harvesting
try:
    for id in tqdm(df_to_read["id"]):
        mask_id = df["id"] == id
        url = "http://export.arxiv.org/abs/" + f"{id}"
        webdata = urllib.request.urlopen(url)
        page = webdata.read().decode("utf-8")
        m = re.search("""<td class="tablecell msc-classes">(.+?)</td>""", page)
        if m != None:
            msc = m.group(1)
            df.loc[mask_id, ["read", "msc"]] = [True, msc]
        else:
            df.loc[mask_id, ["read"]] = [True]
        time.sleep(np.random.randint(40, size=1) / 10)
except:
    print("Something failed, probably connection closed")
    df.to_json(PATH_TO_METADATA_FOLDER + "df_maths_msc.json")
