
### Introduction
This is an ongoing NLP project. This is a classical text classification problem.
The purpose is to construct an automated tool to classify math papers into the math categories of the Arxiv and the MSC classification. 

This is an oingoing project, still in a draft state.  

The dataset can be obtained [on the kaggle page of the Arxiv](https://www.kaggle.com/Cornell-University/arxiv)

This json files is pre-processed with the script creat_metadata.py" to select: 
<ul>
<li> the metadata of the articles published from 2010. </li>
<li> the articles pre-published in a math section only. </li>
This amount to 420017 articles metadata published in the preprint server Arxiv. 
</ul>

The script produces a json file based on the data. The json file contains the section:
<ul>
<li> data of pre-publication</li>
<li> Parsed authors</li>
<li> Title</li>
<li> Abstract</li>
<li> main math category (the primary math category).</li>
<li> all the categories of the papers (primary and if applicable, other categories, including non mathematical categories).</li> 
</ul>

The list of the math labels is: 

'math-ph' 'math.AC' 'math.AG' 'math.AP' 'math.AT' 'math.CA' 'math.CO'  
'math.CT' 'math.CV' 'math.DG' 'math.DS' 'math.FA' 'math.GM' 'math.GN'  
'math.GR' 'math.GT' 'math.HO' 'math.IT' 'math.KT' 'math.LO' 'math.MG'  
'math.MP' 'math.NA' 'math.NT' 'math.OA' 'math.OC' 'math.PR' 'math.QA'  
'math.RA' 'math.RT' 'math.SG' 'math.SP' 'math.ST'

Based on these data, one trains a basic multilabel NLP model consisting of: 
<ul>
<li> a text vectorizer,</li> 
<li> for each label, one trains a linear support vector classifier.</li> 
</ul>
The parameters of the model, as well as the metrics per label, and the global metrics are logged. The data as they exist are widely imbalanced. These data are resampled to make sure the classes are balanced. No data are dropped in the process. 

The vocabulary built for the model is not pre-processed. 

The global metrics are:
<ul>
<li> The average accuracy per class is 0.97.</li> 
<li> The average precision per class is 0.68.</li> 
<li> The average recall per class is 0.57.</li> 
<li> The average F1 per class is 0.62.</li> 
</ul>


### Running
<ol>
<li> Download the data, and store them under  
"data/metatdata/arxiv-metadata-oai-snapshot.json"
</li>
<li> To create the metadata, run "create_metadata_py". This dumps a json "data/df_maths.json" and a csv file "data/df_maths.csv".
</li> To train the model, run "SVC_model.py". The model is dumped as "model/model_SVC.sav"; The metrics are logegd in "log/log_SVC.log".
</li>
</ol>


The file "data_collections.py" was destined to collect the metadata (mathscinet classification data) from the [Arxiv bulk data access](https://arxiv.org/help/bulk_data). Beware of the terms of use of the [Arxiv API](https://arxiv.org/help/api/tou). 


### Metrics per class


'math-ph': {'Accuracy': 0.93, 'Precision:': 0.76, 'Recall': 0.55, 'F1': 0.64},  
'math.AC': {'Accuracy': 0.99, 'Precision:': 0.69, 'Recall': 0.6, 'F1': 0.64},  
'math.AG': {'Accuracy': 0.96, 'Precision:': 0.81, 'Recall': 0.66, 'F1': 0.73},  
'math.AP': {'Accuracy': 0.95, 'Precision:': 0.82, 'Recall': 0.66, 'F1': 0.73},  
'math.AT': {'Accuracy': 0.98, 'Precision:': 0.64, 'Recall': 0.56, 'F1': 0.6},  
'math.CA': {'Accuracy': 0.97, 'Precision:': 0.56, 'Recall': 0.41, 'F1': 0.47},  
'math.CO': {'Accuracy': 0.95, 'Precision:': 0.83, 'Recall': 0.65, 'F1': 0.73},  
'math.CT': {'Accuracy': 0.99, 'Precision:': 0.65, 'Recall': 0.57, 'F1': 0.61},  
'math.CV': {'Accuracy': 0.98, 'Precision:': 0.64, 'Recall': 0.53, 'F1': 0.58},  
'math.DG': {'Accuracy': 0.97, 'Precision:': 0.76, 'Recall': 0.63, 'F1': 0.69},  
'math.DS': {'Accuracy': 0.96, 'Precision:': 0.71, 'Recall': 0.54, 'F1': 0.62},  
'math.FA': {'Accuracy': 0.96, 'Precision:': 0.64, 'Recall': 0.46, 'F1': 0.54},  
'math.GM': {'Accuracy': 0.99, 'Precision:': 0.22, 'Recall': 0.25, 'F1': 0.24},  
'math.GN': {'Accuracy': 0.99, 'Precision:': 0.57, 'Recall': 0.49, 'F1': 0.53},  
'math.GR': {'Accuracy': 0.98, 'Precision:': 0.69, 'Recall': 0.57, 'F1': 0.63},  
'math.GT': {'Accuracy': 0.98, 'Precision:': 0.72, 'Recall': 0.63, 'F1': 0.67},  
'math.HO': {'Accuracy': 0.99, 'Precision:': 0.53, 'Recall': 0.44, 'F1': 0.48},  
'math.IT': {'Accuracy': 0.98, 'Precision:': 0.92, 'Recall': 0.81, 'F1': 0.86},  
'math.KT': {'Accuracy': 0.99, 'Precision:': 0.49, 'Recall': 0.4, 'F1': 0.44},  
'math.LO': {'Accuracy': 0.99, 'Precision:': 0.82, 'Recall': 0.72, 'F1': 0.77},  
'math.MG': {'Accuracy': 0.98, 'Precision:': 0.5, 'Recall': 0.48, 'F1': 0.49},  
'math.MP': {'Accuracy': 0.93, 'Precision:': 0.76, 'Recall': 0.55, 'F1': 0.64},  
'math.NA': {'Accuracy': 0.97, 'Precision:': 0.82, 'Recall': 0.71, 'F1': 0.76},  
'math.NT': {'Accuracy': 0.97, 'Precision:': 0.81, 'Recall': 0.66, 'F1': 0.73},  
'math.OA': {'Accuracy': 0.99, 'Precision:': 0.68, 'Recall': 0.6, 'F1': 0.64},  
'math.OC': {'Accuracy': 0.96, 'Precision:': 0.83, 'Recall': 0.69, 'F1': 0.75},   
'math.PR': {'Accuracy': 0.96, 'Precision:': 0.8, 'Recall': 0.66, 'F1': 0.72},  
'math.QA': {'Accuracy': 0.98, 'Precision:': 0.55, 'Recall': 0.52, 'F1': 0.54},  
'math.RA': {'Accuracy': 0.98, 'Precision:': 0.58, 'Recall': 0.5, 'F1': 0.53},  
'math.RT': {'Accuracy': 0.97, 'Precision:': 0.7, 'Recall': 0.58, 'F1': 0.63},  
'math.SG': {'Accuracy': 0.99, 'Precision:': 0.69, 'Recall': 0.58, 'F1': 0.63},  
'math.SP': {'Accuracy': 0.99, 'Precision:': 0.51, 'Recall': 0.5, 'F1': 0.5},  
'math.ST': {'Accuracy': 0.98, 'Precision:': 0.75, 'Recall': 0.71, 'F1': 0.73}  