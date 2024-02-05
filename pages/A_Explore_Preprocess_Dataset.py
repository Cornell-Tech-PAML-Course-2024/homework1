import streamlit as st                  
import pandas as pd
import plotly.express as px
from pandas.plotting import scatter_matrix
from itertools import combinations
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

feature_lookup = {
    'longitude':'**longitude** - longitudinal coordinate',
    'latitude':'**latitude** - latitudinal coordinate',
    'housing_median_age':'**housing_median_age** - median age of district',
    'total_rooms':'**total_rooms** - total number of rooms per district',
    'total_bedrooms':'**total_bedrooms** - total number of bedrooms per district',
    'population':'**population** - total population of district',
    'households':'**households** - total number of households per district',
    'median_income':'**median_income** - median income',
    'ocean_proximity':'**ocean_proximity** - distance from the ocean',
    'median_house_value':'**median_house_value**'
}

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

# Checkpoint 1
def load_dataset(filepath):
    """
    This function uses the filepath (string) a .csv file locally on a computer 
    to import a dataset with pandas read_csv() function. Then, store the 
    dataset in session_state.

    Input: data is the filename or path to file (string); open file using pd.read_csv
    Output: pandas dataframe df
    """
    data=None
    return data

# Helper Function
def display_features(df,feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup and dataset columns).
    
    Inputs:
    - df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
    - feature_lookup (dict): A dictionary containing the descriptions for the features.
    
    Outputs: None
    """
    for idx, col in enumerate(df.columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s'%(idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s'%(idx, col))

# Checkpoint 2
def sidebar_filter(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar 

    Input: 
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output: 
        - list of sidebar filters on features
    """
    side_bar_data = []
    return side_bar_data

# Checkpoint 3
def summarize_missing_data(df, top_n=3):
    """
    This function summarizes missing values in the dataset

    Input: 
        - df: the pandas dataframe
        - top_n: top n features with missing values, default value is 3
    Output: 
        - a dictionary containing the following keys and values: 
            - 'num_categories': counts the number of features that have missing values
            - 'average_per_category': counts the average number of missing values across features
            - 'total_missing_values': counts the total number of missing values in the dataframe
            - 'top_missing_categories': lists the top n features with missing values
    """
    out_dict = {'num_categories': 0,
                'average_per_category': 0,
                'total_missing_values': 0,
                'top_missing_categories': []}
    return out_dict

# Checkpoint 4
def remove_features(df,removed_features):
    """
    Remove the features in removed_features (list) from the input pandas dataframe df. 

    Input: df is dataset in pandas dataframe
    Output: pandas dataframe df
    """
    return df

# Checkpoint 5
def impute_dataset(df, impute_method):
    """
    Impute the dataset df with imputation method impute_method 
    including mean, median, zero values or drop Nan values in 
    the dataset (all numeric and string columns).

    Input: 
    - df is dataset in pandas dataframe
    - impute_method = {'Zero', 'Mean', 'Median','DropNans'}
    Output: pandas dataframe df
    """
    return df

# Checkpoint 6
def one_hot_encode_feature(df, feature):
    """
    This function performs one-hot-encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform one-hot-encoding
    Output: 
        - df: dataframe with one-hot-encoded feature
    """
    return df

# Checkpoint 7
def integer_encode_feature(df, feature):
    """
    This function performs integer-encoding on the given features

    Input: 
        - df: the pandas dataframe
        - feature: the feature(s) to perform integer-encoding
    Output: 
        - df: dataframe with integer-encoded feature
    """
    return df

# Checkpoint 8
def scale_features(df, features, scaling_method): 
    """
    Use the scaling_method to transform numerical features in the dataset df. 

    Input: 
        - Features X
        - Scaling method is a string; Options include {'Standardarization', 'Normalization', 'Log'}
    Output: 
        - Standarization: X_new = (X - mean)/Std
        - Normalization: X_new = (X - X_min)/(X_max - X_min)
        - Log: X_new = log(X)
    """
    return df

# Checkpoint - 9
def create_feature(df, math_select, math_feature_select, new_feature_name):
    """
    Create a new feature with name new_feature_name in dataset df with the 
    mathematical operation math_select (string) on features math_feature_select (list). 

    Input: 
        - df: the pandas dataframe
        - math_select: the math operation to perform on the selected features
        - math_feature_select: the features to be performed on
        - new_feature_name: the name for the new feature
    Output: 
        - df: the udpated dataframe
    """
    return df

# Checkpoint - 10
def remove_outliers(df, feature, outlier_removal_method=None):
    """
    This function removes the outliers of the given feature(s)

    Input: 
        - df: pandas dataframe
        - feature: the feature(s) to remove outliers
    Output: 
        - dataset: the updated data that has outliers removed
        - lower_bound: the lower 25th percentile of the data
        - upper_bound: the upper 25th percentile of the data
    """
    lower_bound = 0
    upper_bound = 0
    return df, lower_bound, upper_bound

## Checkpoint - 11
def compute_descriptive_stats(df, stats_feature_select, stats_select):
    """
    Compute descriptive statistics stats_select on a feature stats_feature_select 
    in df. Statistics stats_select include mean, median, max, and min. Return 
    the results in an output string out_str and dictionary out_dict (dictionary).

    Input: 
    - df: the pandas dataframe
    - stats_feature_select: list of feaures to computer statistics on
    - stats_select: list of mathematical opations
    Output: 
    - output_str: string used to display feature statistics
    - out_dict: dictionary of feature statistics
    """
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    return output_str, out_dict

# Checkpoint 12
def compute_correlation(df, features):
    """
    This function computes pair-wise correlation coefficents of X and render summary strings

    Input: 
        - df: pandas dataframe 
        - features: a list of feature name (string), e.g. ['age','height']
    Output: 
        - correlation: correlation coefficients between one or more features
        - summary statements: a list of summary strings where each of it is in the format: 
            '- Features X and Y are {strongly/weakly} {positively/negatively} correlated: {correlation value}'
    """
    correlation = 0
    cor_summary_statements = []
    return correlation, cor_summary_statements

###################### FETCH DATASET #######################

# Use file_uploader to upload the dataset locally
df=None

######################### MAIN BODY #########################

######################### EXPLORE DATASET #########################

if df is not None:
    st.markdown('### 1. Explore Dataset Features')

    # Display feature names and descriptions (from feature_lookup)
    display_features(df,feature_lookup)
    
    # Display dataframe as table
    st.dataframe(df)
    X = df.copy()

    ###################### VISUALIZE DATASET #######################
    st.markdown('### 2. Visualize Features')

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots','Lineplots','Histogram','Boxplot']
    )

    # Draw plots
    if chart_select == 'Scatterplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.scatter(data_frame=df,
                                x=x_values, y=y_values,
                                range_x=[side_bar_data[0][0],
                                        side_bar_data[0][1]],
                                range_y=[side_bar_data[1][0],
                                        side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Histogram':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.histogram(data_frame=df,
                                x=x_values,
                                range_x=[side_bar_data[0][0],
                                            side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Lineplots':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            side_bar_data = sidebar_filter(
                df, chart_select, x=x_values, y=y_values)
            plot = px.line(df,
                            x=x_values,
                            y=y_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]],
                            range_y=[side_bar_data[1][0],
                                    side_bar_data[1][1]])
            st.write(plot)
        except Exception as e:
            print(e)
    if chart_select == 'Boxplot':
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            side_bar_data = sidebar_filter(df, chart_select, x=x_values)
            plot = px.box(df,
                            x=x_values,
                            range_x=[side_bar_data[0][0],
                                    side_bar_data[0][1]])
            st.write(plot)
        except Exception as e:
            print(e)

    #st.sidebar.button("Reset", type="primary")
    if st.sidebar.button('Clip feature from %.2f to %.2f' % (side_bar_data[0][0], side_bar_data[0][1])):
        df[x_values+'_clipped'] = df[x_values]
        df[df[x_values+'_clipped']>side_bar_data[0][1]] = 0
        df[df[x_values+'_clipped']<side_bar_data[0][0]] = 0
        st.sidebar.write(x_values + ' cliped from '+str(side_bar_data[0][0])+' to '+str(side_bar_data[0][1]))
        if(chart_select == 'Scatterplots' or chart_select == 'Lineplots'):
            df[y_values+'_clipped'] = df[y_values]
            df[df[y_values+'_clipped']>side_bar_data[1][1]] = 0
            df[df[y_values+'_clipped']<side_bar_data[1][0]] = 0
            st.sidebar.write(y_values + ' cliped from '+str(side_bar_data[1][0])+' to '+str(side_bar_data[1][1]))

    # Display original dataframe
    st.markdown('## 3. View initial data with missing values or invalid inputs')
    st.dataframe(df)

    X = df.copy()
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Show summary of missing values including 
    missing_data_summary = summarize_missing_data(df)

    # Remove param
    st.markdown('### 4. Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        df.columns,
    )
    df = remove_features(df, removed_features)

    ########
    # Display updated dataframe
    st.dataframe(df)

    # Impute features
    st.markdown('### 5. Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    impute_method = st.selectbox(
        'Select imputation method',
        ('Zero', 'Mean', 'Median','DropNans')
    )

    # Call impute_dataset function to resolve data handling/cleaning problems
    df = impute_dataset(df, impute_method)
    
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.dataframe(df)

    ############################################# PREPROCESS DATA #############################################

    # Handling Text and Categorical Attributes
    st.markdown('### 6. Handling Text and Categorical Attributes')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_feature_select_int = st.selectbox(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            df = integer_encode_feature(df, text_feature_select_int)
    
    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            df = one_hot_encode_feature(df, text_feature_select_onehot)

    # Show updated dataset
    st.write(df)

    # Sacling features
    st.markdown('### 7. Feature Scaling')
    st.markdown('Use standardarization or normalization to scale features')

    # Use selectbox to provide impute options {'Standardarization', 'Normalization', 'Log'}
    scaling_method = st.selectbox(
        'Select feature scaling method',
        ('Standardarization', 'Normalization', 'Log')
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    scale_features_select = st.multiselect(
        'Select features to scale',
        numeric_columns,
    )

    # Call scale_features function to scale features
    df = scale_features(df, scale_features_select, scaling_method)
    #########

    # Display updated dataframe
    st.dataframe(df)

    # Create New Features
    st.markdown('## 8. Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        df = create_feature(
                            df, math_select, math_feature_select, new_feature_name)
                        st.write(df)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    df = create_feature(df, math_select, [
                                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(df)

    st.markdown('### 9. Inspect Features for outliers')
    outlier_feature_select = None
    numeric_columns = list(df.select_dtypes(include='number').columns)

    outlier_method_select = st.selectbox(
        'Select statistics to display',
        ['IQR', 'STD']
    )

    outlier_feature_select = st.selectbox(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df, lower_bound, upper_bound = remove_outliers(
            df, outlier_feature_select, outlier_method_select)
        st.write('Outliers for feature %s are lower than %.2f and higher than %.2f' % (
            outlier_feature_select, lower_bound, upper_bound))
        st.write(df)

    # Descriptive Statistics 
    st.markdown('### 10. Summary of Descriptive Statistics')

    stats_numeric_columns = list(df.select_dtypes(['float','int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )

    stats_select = st.multiselect(
        'Select statistics to display',
        ['Mean', 'Median','Max','Min']
    )
            
    # Compute Descriptive Statistics including mean, median, min, max
    display_stats, _ = compute_descriptive_stats(df, stats_feature_select, stats_select)

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### 11. Correlation Analysis")

    # Collect features for correlation analysis using multiselect
    #df = st.session_state['house_df']
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)