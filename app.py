# region Load Madule
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report

# endregion
# region Function Declare
# region def Load_data


@st.cache_data
def load_data():
    # columns
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    # read files
    df = pd.read_csv('data/car.csv', names=columns, header=0 , sep=";")
    return df
# endregion
#endregion
df = load_data()
# region  Streamlit Page
st.set_page_config(page_title="Car Evaluation Predictor",layout='wide')
st.title("Predict Car Evaluation")
st.markdown('This Program Predict the acceptability of a car based on its features.')
with st.expander('View Data'):
    st.dataframe(df.head(10))
    st.write('Total Row: ',len(df))
    st.write('Columns: ',df.columns.tolist())
    st.write('Goal Variable Distribute:')
    st.bar_chart(df['class'].value_counts())
# endregion



