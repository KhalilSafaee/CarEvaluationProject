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

# region Train,Test set
# one-hot encoding
X = pd.get_dummies(df.drop('class',axis = 1))
y = df['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
dt_model = DecisionTreeClassifier(max_depth=5,random_state=42)
dt_model.fit(X_train,y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
# endregion

st.metric("Model accuracy" ,f"{accuracy:.2%}")
st.header('Predict for New Car')
col1 , col2,col3 = st.columns(3)
with col1:
    buying = st.selectbox('Buy Price Category',['low','med','high','vhigh'])
    maint = st.selectbox('maintenance Cost Category',['low','med','high','vhigh'])
with col2:
    doors = st.selectbox('Doors Count',['2','3','4','5more'])
    persons = st.selectbox('Person Capacity',['2','4','more'])






