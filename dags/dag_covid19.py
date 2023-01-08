import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

from airflow import DAG
from datetime import datetime,timedelta
import sqlite3
import csv
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

from src.streamlit_ys import covid_ML_strm

default_args = {
    'owner': 'yogasugitha',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}


def covid_ETL(pathcsv, pathdb):
    #extract data from csv file
    print('loading data from %s' % pathcsv)
    df = pd.read_csv(pathcsv)

    #transform data
    ##convert DATE_DIED to binary DEATH
    cols = ['PNEUMONIA','DIABETES', 'COPD', 'ASTHMA', 'INMSUPR','HIPERTENSION', 
        'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY','RENAL_CHRONIC', 'TOBACCO']
    for col in cols :
        df = df[(df[col] == 1)|(df[col] == 2)]
    df['DEATH'] = [2 if row=='9999-99-99' else 1 for row in df['DATE_DIED']]
    ##drop column
    df.drop(columns=['INTUBED','ICU','DATE_DIED'],inplace=True)
    print('MALE',df.query('SEX==2')['PREGNANT'].value_counts(), sep='\n')
    print('FEMALE',df.query('SEX==1')['PREGNANT'].value_counts(), sep='\n')
    df['PREGNANT'] = df['PREGNANT'].replace(97,2)
    df['PREGNANT'] = df['PREGNANT'].replace(98,2)
    
    #load to db sqlite3
    db = sqlite3.connect(pathdb)
    cursor = db.cursor()

    df.to_sql(name='yoga_cv',con=db,if_exists='replace', index=False)
    db.commit()
    cursor.close()
    db.close()         
    print('!!!ETL DONE!!!')


def covid_visualize(path):
    print('loading data from %s' % path)
    con = sqlite3.connect(path)
    df = pd.read_sql_query('select * from yoga_cv', con)

    plt.figure(figsize=(18,15))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.title('Correlation Between Features', fontsize=18)
    file_path = 'dags/src/image/corr_heatmap.png'
    plt.savefig(file_path)
    print('!!!!Visualization Done!!!')

def covid_ML(path):
    print('loading data from %s' % path)
    con = sqlite3.connect(path)
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
    print("Logistic Regression Accuracy :", log_reg.score(test_x, test_y))

    print("Logistic Regression F1 Score : ", f1_score(test_y, log_reg.predict(test_x),average=None))

    sns.heatmap(confusion_matrix(test_y, log_reg.predict(test_x)), annot=True, fmt='.0f')
    plt.title("Logistic Regression Confusion Matrix", fontsize=18)
    file_path = 'dags/src/image/confusion_matrix.png'
    plt.savefig(file_path)
    print('!!!Machine Learning Completed Successfully!!!')
    


with DAG(
    default_args=default_args,
    dag_id = 'dag_covidYS',
    description='dag untuk scheduling task data engineering dengan tema covid 19',
    start_date=datetime(2022,12,29),
    schedule_interval='@daily'
) as dag:
    task_ETL = PythonOperator(
        task_id='data_covid_etl',
        dag=dag,
        python_callable=covid_ETL,
        op_kwargs={
            'pathcsv': 'dags/src/data/CovidData.csv',
            'pathdb' : 'dags/src/data/yoga_cvDB.db',

        }
    )
    task_VIZ = PythonOperator(
        task_id='data_covid_visualization',
        dag=dag,
        python_callable=covid_visualize,
        op_kwargs={
            'path': 'dags/src/data/yoga_cvDB.db',

        }
    )
    task_ML= PythonOperator(
        task_id='data_covid_machine_learning',
        dag=dag,
        python_callable=covid_ML,
        op_kwargs={
            'path': 'dags/src/data/yoga_cvDB.db',

        }
    )
    task_strm = PythonOperator(
        task_id='data_covid_strm',
        dag=dag,
        python_callable=covid_ML_strm,
    )
    task_ETL >> [task_VIZ, task_strm] >> task_ML 





