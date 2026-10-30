## Credit Card Defaulters Machine Learning algorithms Project

## Created virtual environment
```
CreditCardDefaulters
Python version 3.11
conda activate CreditCardDefaulters/
```
### Install all necessary libraries
```
pip install -r requirements.txt
```

### Introduction
```
The aim of this study is to use some supervised machine learning algorithms to identify the key drivers that determine the likelihood of credit card default. Credit card default happens when you have become severely delinquent on your credit card payments.

The goal is to build an automated model for both identifying the key factors, and predicting a credit card default based on the information about the client and historical transactions.
```

### Dataset information
```
The dataset used in this study is the Default of credit card clients from the UCI machine learning repository.
It consists of 30000 observations that represent distinct credit card clients. Each observation has 24 attributes that contain information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients

The first group of variables contains information about the client personal information:

ID: ID of each client, categorical variable
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
SEX: Gender, categorical variable (1=male, 2=female)
EDUCATION: level of education, categorical variable (0=unknown,1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status, categorical variable (0=unknown,1=married, 2=single, 3=others)
AGE: Age in years, numerical variable
The following attributes contains information about the delay of the past payment referred to a specific month:

PAY_0: Repayment status in September 2005 (-2=No consumption/no credit use for that period,-1=pay duly,0=Use of revolving credit: a balance is carried forward, 1=payment delay for one month, 2=payment delay for two months,3,4,...=Corresponding number of months of payment delay)
PAY_2: Repayment status in August 2005 (same scale as before)
PAY_3: Repayment status in July 2005 (same scale as before)
PAY_4: Repayment status in June 2005 (same scale as before)
PAY_5: Repayment status in May 2005 (same scale as before)
PAY_6: Repayment status in April 2005 (same scale as before)

Other variables instead consider the information related to the amount of bill statement (i.e. a monthly report that credit card companies issue to credit card holders in a specific month):

BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)

The following variables instead consider the amount of previous payment in a specific month:

PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)

The last variable is the one to be predicted:

default payment next month: indicate whether the credit card holders are defaulters or non-defaulters (1=yes, 0=no)
```
