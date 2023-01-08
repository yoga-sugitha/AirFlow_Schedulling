# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.model_selection import train_test_split
from PIL import Image
def covid_ML_strm():
    con = sqlite3.connect('dags/src/data/yoga_cvDB.db')
    df = pd.read_sql_query('select * from yoga_cv', con)

    #scale numerical feature
    standard_scaler = StandardScaler()
    df['AGE'] = standard_scaler.fit_transform(df.loc[:,['AGE']])

    #menentukan X, Y data
    y = df['DEATH']
    x = df.drop(['DEATH'], axis=1)
    #spliting data test and training
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)


    #logistic regression algorithm
    log_reg = LogisticRegression()
    log_reg.fit(train_x, train_y)

    # Plot the confusion matrix
    plt.matshow(confusion_matrix(test_y, log_reg.predict(test_x)))
    plt.title("Logistic Regression Confusion Matrix", fontsize=18)

    # Display the plot in the dashboard
    st.pyplot()

    # Use Streamlit to create a dashboard
    st.title("Covid-19 Machine Learning Results")

    # Create a sidebar on the left side of the dashboard
    st.sidebar.title("Covid-19 Machine Learning Results")

    # Add some formatting to the text
    st.markdown("### Results")

    # Display the results in the sidebar
    accuracys = log_reg.score(test_x, test_y)
    st.sidebar.write("Accuracy: ", accuracys)

    f1_scores = f1_score(test_y, log_reg.predict(test_x),average=None)
    st.sidebar.write("F1 Score: ", f1_scores)

# Add a run button
if st.button("Run"):
    covid_ML_strm()

