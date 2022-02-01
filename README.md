This is an ongoing NLP project. This is a classical text classification problem.
The purpose is to construct an automated tool to classify math papers into the math categories of the Arxiv
 and the MSC classification. 

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
<li> Asbtract</li>
<li> main math category (the primary math categories)</li>
<li> all the categories of the papers (primiary and if applicable, other categories, including not mathematical categories).</li> 
</ul>

The list of the labels is: 
'math-ph' 'math.AC' 'math.AG' 'math.AP' 'math.AT' 'math.CA' 'math.CO'
'math.CT' 'math.CV' 'math.DG' 'math.DS' 'math.FA' 'math.GM' 'math.GN'
'math.GR' 'math.GT' 'math.HO' 'math.IT' 'math.KT' 'math.LO' 'math.MG'
'math.MP' 'math.NA' 'math.NT' 'math.OA' 'math.OC' 'math.PR' 'math.QA'
'math.RA' 'math.RT' 'math.SG' 'math.SP' 'math.ST'

Based on these data, one trains a basic multilabel NLP model consisting of: 
<ul>
<li> a text vectorizer,</li> 
<li> for each label, one trains a Linear Support Vector Classifier.</li> 
</ul>
The parameters of the model, as well as the metrics per label, and the global metrics are logged. 

The vocabulary built for the model is not pre-processed. 

The metrics per label are:



The global metrics are:










