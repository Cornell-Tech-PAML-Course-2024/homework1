# Practical Applications in Machine Learning - Homework 1

Homework 1: The goal of this assignment is to perform data preprocessing on the California Housing dataset. The learning outcomes for this assignment are:
- Learn when to use Standardization vs Normalization
- Learn to handle missing data
- Apply feature encoding to appropriate data types
- Apply outlier detection methods using Interquartile and standard deviation methods
- Create new features using feature engineering
- Compute correlation coefficients and interpret the results

This assignment is contains two parts:

- <b>Programming:<b> Write a web application that enables users to upload a dataset, visualize the data, and perform preprocessing steps to gain useful insights for ML.
- <b>Online Assessment:<b> Respond to reflection questions to assess your understanding of preprocessing steps. 

* <b>Due</b>: February 21, 2024 at 11:00PM 
* <b>What to turn in</b>: Submit responses on GitHub AutoGrader
* <b>Assignment Type</b>: Groups (Up to 3)
* <b>Time Estimate</b>: 9 Hours
* <b>Submit code via GitHub
* <b>Submit Reflection Assessment</b> via Canvas (9 questions)

<p align="center"> 
<img src="./images/explore_data_hw1.gif" width="70%"> 
<i>

<b>Figure:</b> This shows a demonstration of the web application for End-to-End ML pipelines.

# Installation

Install [Streamlit](https://streamlit.io/)
```
pip install streamlit     # Install streamlit
streamlit hello           # Test installation
```

```
Install other requirements

```
pip install numpy
pip install pandas
pip install plotly
pip install itertools
pip install sklearn-learn
```

* preprocess_data.py: HW1 assignment template using streamlit for web application UI and workflow of activties. 
* pages/*.py files: Contains code to explore data, preprocess it and prepare it for ML. It includes checkpoints for the homework assignment.
* datasets: folder that conatins the dataset used for HW1 in 'housing/housing.csv'
* notebooks: jupyter notebook on preprocessing steps.
* test_homework1.py: contains Github autograder functions

## 1.1 California Housing Dataset

Create useful visualizations for machine learning tasks. This assignment focuses on visualizing features from a dataset given some input .csv file (locally or in the cloud), the application is expected to read the input dataset. Use the pandas read_csv function to read in a local file. Use Streamlit layouts to provide multiple options for interacting with and understanding the dataset.

This assignment involves testing the end-to-end pipeline in a web application using a California Housing dataset from the textbook: Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O’Reilly Media, Inc., 2022 [[GitHub](https://github.com/ageron/handson-ml2)]. The dataset was capture from California census data in 1990 and contains the following features:
* longitude - longitudinal coordinate
* latitude - latitudinal coordinate
* housing_median_age - median age of district
* total_rooms - total number of rooms per district
* total_bedrooms - total number of bedrooms per district
* population - total population of district
* households - total number of households per district'
* median_income - median income
* ocean_proximity - distance from the ocean
* median_house_value - median house value

## 1.2 Explore dataset (see HW1 document)

## 1.3 Preprocess data (see HW1 document)

## 1.4 Testing Code with Github AutoGrader
```
streamlit run preprocess_data.py
```
## Test code using pytest by running the test_homework1.py file (see below). There are 6 test cases, one for each checkpoint above.
```
pytest
```

## Run homework assignment web application
```
cd $HOME # or whatever development directory you chose earlier
cd homework1 # go to this project's directory
streamlit run preprocess_data.py
```

# 2. Reflection Assessment

Submit on Canvas.

# Further Issues and questions ❓

If you have issues or questions, don't hesitate to contact the teaching team:

* Angelique Taylor (amt298@cornell.edu) - Instructor
* Tauhid Tanjim (tt485@cornell.edu) - Teaching Assistant
* Olga Esperanza Acuna Leanos (oea9@cornell.edu) - Grader
* Kexin Cheng (kc2248@cornell.edu) - Grader