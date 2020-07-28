# Data Science Salary Estimator: Project Overview 
* Created a tool that estimates data science salaries to help data scientists negotiate their income when they get a job.
* Scraped over 500 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark. 
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium 
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905  

## Web Scraping
Tweaked the web scraper github repo (above) to scrape 1000 job postings from glassdoor.com. With each job, we got the following:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
After scraping the data, cleaned it up so that it was usable for our model. I did this by making following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state, if the state was not mentioned I used company’s headquarters
*	Transformed founded date into company age 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/sid123github/DS_Salary_Predictor/blob/master/EDA_Images/Jobs_by_Location.PNG "Job Opportunities by State")
![alt text](https://github.com/sid123github/DS_Salary_Predictor/blob/master/EDA_Images/top_Companies.PNG "Top Hiring Companies")

* I also build a Word Cloudd from all the Job Description to understand most common requirements

![alt text](https://github.com/sid123github/DS_Salary_Predictor/blob/master/EDA_Images/WordCloud_JobDescription.PNG "Word Cloud for Job Descriptions")

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 33%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 51.22
*	**Lasso Regression**: MAE = 58.34
*	**Linear Regression**: MAE = 69.26
