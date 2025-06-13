#hamtienich
import streamlit as st 
import pandas as pd

@st.cache_data
def load_data():
    gender_submission_df = pd.read_csv('data/gender_submission.csv')
    test_df = pd.read_csv('data/test.csv')
    train_df = pd.read_csv('data/train.csv')
    
    return gender_submission_df, test_df, train_df


def preprocess_data(df):
    df = df.copy()
    df['Age'] = df['Age'].fillna(-1)
    df['AgeGroup'] = pd.cut(
        df['Age'], 
        bins=[-1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 100], 
        labels=['Unknown','0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-100']
    )
    return df

