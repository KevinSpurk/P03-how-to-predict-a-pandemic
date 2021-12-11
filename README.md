![alt text](https://github.com/KevinSpurk/P03-how-to-predict-a-pandemic/blob/main/presentation/masked_crowd.png "Covid title graphic")



# P03 | Prediction possibilities in a pandemic

### Introduction

In this analysis I'm investigating the Covid 19 pandemic and have a look at the prediction opportunities of specific metrics such as the number of new daily cases and new daily deaths. The goal is to find out how well you can predict those metrics for a date in the future based on past and present data, comparing the prediction performance of two different datasets. One provides data of peoples movement patterns in areas such as retail, parks, public transport, etc. The other reflects search patterns of keywords connected to the pandemic and someone with a potential Covid infection experiencing symptoms might look for online. The I’m using data of 13 European countries to allow for a cross-country and a by-country analysis and prediction.



# Table of content

**1. Setup**

Importing libraries and datasets

**2. Data wrangling**

Initial elimination of  unnecessary features, data cleaning and wrangling.

**3. EDA**

Statistical and visual exploration of the data.

**4. Data Preprocessing**

Feature selection and using techniques like encoding and time lags to prepare the data for modeling.

**5. Baseline model**

Creating a first prediction model based on the mobility data and evaluate its performance to compare against further improvements and alternative model algorithms.

**6. Mobility data model**

Developing the prediction model based on the mobility data. Applying feature transformations, using hyper parameter tuning and different model algorithms and comparing the results of different approaches.

**7. Search trends data model**

Developing the prediction model based on the search trends data equivalent to the previous step.

**8. Model with combined data**

Checking if a model using both data sets can have better prediction performance.

**9. Conclusions**

A comparative summary of the modelling results and a recommendation for next step to take when trying to predict the metrics of the Covid 19 pandemic.



# Data sources:

#### Google LLC "Google COVID-19 Community Mobility Reports"

https://www.google.com/covid19/mobility/ Accessed: <date>.

#### Data on COVID-19 (coronavirus) by Our World in Data
  
https://github.com/owid/covid-19-data/

Autors: This data has been collected, aggregated, and documented by Cameron Appel, Diana Beltekian, Daniel Gavrilov, Charlie Giattino, Joe Hasell, Bobbie Macdonald, Edouard Mathieu, Esteban Ortiz-Ospina, Hannah Ritchie, Lucas Rodés-Guirao, Max Roser.
  

#### COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University

https://github.com/CSSEGISandData/COVID-19
  
