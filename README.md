# 🐤 JPM_SMM4H_2b_LW

**Subtask 2b** for the **7th Social Mining Media for Health (SMM4H)** competition hosted by **International Conference on Computational Logistics (COLING) 2022**

_by Leung Wai Liu_

This repo is training, ensembling and analysis code for the BERT Model used for Subtask 2b of the SMM4H competition that I competed in during my internship at the AI Research team in J.P. Morgan Chase in the Summer of 2022. 

_\#NLP \#BERT \#ML \#Python_

**See Also:** Subtask 1a | Subtask 1b | Subtask 1c | Subtask 2a | Subtask 5 

## Premise
The need to use Natural Language Processing \(NLP\) on social media posts is increasingly important as its userbase grows to guage public perception on issues, such as sentiments during the COVID-19 pandemic. 

## Task Description
Task 2 is a health mandate related task, which Subtask 2b is a premise classification task consisting of classifying whether a tweet has a premise pertaining to the particular label.

## Methodology
The datasets were trained on variants top of the BERT language model \(Devlin et al., 2019\): RoBERTa<sub>BASE</sub>, RoBERTa<sub>LARGE</sub>, BERT<sub>BASE</sub>-uncased, BERT<sub>LARGE</sub>-uncased.

The model ensembling methodology is adapted from \(Jayanthi and Gupta, 2021\) method of model ensembling. Various methods of ensembling were experimented, including majority-vote, weighted and unweighted. Ultimately, a weighted average ensemble of RoBERTa<sub>LARGE</sub> models were used. 

## Results 
**Performance Metric for Subtask 1a**
| Task | F1-Score |
| ---: | :---: |
| Task 2b | 0.701 | 

> Tied **1st** of 14 submissions

## Special Thanks
- **Akshat Gupta**, for being a great project manager and guiding us through NLP from start to finish
- **Saheed Obitayo**, for being a great manager
- **J.P. Morgan AI Research** and **Prep for Prep** for the incredible opportunity for the internship
- The organizers for the 7th SMM4H competition and 2022 COLING conference
