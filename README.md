# Practical Applications in Machine Learning - Homework 1

The goal of Homework 1 assignment is to build your first end-to-end Machine Learning (ML) pipeline using public datasets and by creating your own datasets. The <b>learning outcomes</b> for this assignment are: 

* Build framework for end-to-end ML pipeline in Streamlit. Create your first web application! 
* Develop web application that walks users through steps of ML pipeline starting with data visualization and preprocessing steps. 

This assignment is contains two parts:

1. <b>End-to-End ML Pipeline</b>: Many ML projects are NOT used in production or NOT easily used by others including ML engineers interested in exploring prior ML models, testing their models on new datasets, and helping users explore ML models. To address this challenge, the goal of this assignment is to implement a front- and back-end ML project, focusing on the dataset exploration and preprocessing steps. The hope is that the homework assignments help students showcase ML skills in building end-to-end ML pipelines and deploying ML web applications for users which we build on in future assignments. 

2. <b>Dataset Curation (In-Class Activity)</b>: It is often challenging to collect datasets when none exists in your problem domain. Thus, it is important to understand how to curate new datasets and explore existing methodologies for data collection. Part II of HW1 focuses on how to collect datasets, annotate the data, and evaluate the annotations in preparation for ML tasks.

HW1 serves as an outline for the remaining assignments in the course, building end-to-end ML pipelines and deploying useful web applications using those models. This assignment in particular focuses on the data exploration and preprocess. 

* <b>Due</b>:  Friday February 17, 2023 at 11:00PM 
* <b>What to turn in</b>: Submit responses on GitHub AutoGrader
* <b>Assignment Type</b>: Individual
* <b>Time Estimate</b>: 9 Hours
* <b>Submit code via GitHub</b>: https://classroom.github.com/a/fiL30jIe
* <b>Submit Reflection Assessment</b> via Canvas (multiple choice, 5 questions)

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
```

* preprocess_data.ipynb: This is the example from the textbook on predicting housing prices. We will use this notebook to create an online ML end-to-end pipeline. We will focus on data collction and preprocessing steps.
* preprocess_data.py: HW1 assignment template using streamlit for web application UI and workflow of activties. 
* pages/*.py files: Contains code to explore data, preprocess it and prepare it for ML. It includes checkpoints for the homework assignment.
* datasets: folder that conatins the dataset used for HW1 in 'housing/housing.csv'
* notebooks: contains example notebooks for HW1
* test_homework1.py: contains Github autograder functions
* images/: contains images for readme

# 1. Build End-to-End ML Pipeline

The first part of HW1 focuses on ‘Building an End-to-End ML Pipeline’ which consists of creating modules that perform the following tasks: exploring and visualizing the data to gain insights and preprocess and prepare data for machine learning algorithms.

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

## 1.2 Explore dataset

## 1.3 Preprocess data

## 1.4 Testing Code with Github AutoGrader

* Create GitHub Account (if not already done)

* Submit an email address to the teaching staff to connect your account for grading: https://forms.gle/7m7xAKcTv6DdE1E98

* Github Submission: https://classroom.github.com/a/fiL30jIe

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

Submit on Gradescope.

# Further Issues and questions ❓

If you have issues or questions, don't hesitate to contact the teaching team:

* Angelique Taylor (amt298@cornell.edu) - Instructor
* Tauhid Tanjim (tt485@cornell.edu) - Teaching Assistant
* Jinzhao Kank (jk2575@cornell.edu) - Grader
* Kathryn Gdula (kg435@cornell.edu) - Grader