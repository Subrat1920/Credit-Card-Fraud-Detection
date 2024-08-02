import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

credit_card_df = pd.read_csv('creditcard.csv')
credit_card_df.head()

"""
Time column defines the time lapse of the credit card transaction
V1-V28 are the features of all the credit cards
Amount is the credit creadibile 
Class defines whether the transaction is legitimate (0) or illegitimate transaction (1)
"""

legit = credit_card_df[credit_card_df.Class == 0]
fraud = credit_card_df[credit_card_df.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)

credited_card_df = pd.concat([legit_sample, fraud], axis=0)

credited_card_df["Class"].value_counts()

credited_card_df.groupby('Class').mean()

x = credited_card_df.drop('Class', axis=1)
y = credited_card_df['Class']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)

train_acc = accuracy_score(model.predict(x_train), y_train)
test_acc = accuracy_score(model.predict(x_test), y_test)


st.title("Credit Card Fraud Detection Model")
input_df = st.text_input("Enter All Required Features Values")

input_df_splited = input_df.split(',')

submit = st.button("Submit")

if submit:
    try:
        # Splitting and filtering the input values
        input_df_splited = [x for x in input_df.split(',') if x.strip()]

        # Validate the number of input features
        if len(input_df_splited) != x.shape[1]:
            st.write(f"Please enter exactly {x.shape[1]} feature values.")
        else:
            # Convert input values to a numpy array of floats
            features = np.asarray(input_df_splited, dtype=np.float64)

            # Reshape and make prediction
            prediction = model.predict(features.reshape(1, -1))

            if prediction[0] == 0:
                st.write("Legitimate Transaction")
            else:
                st.write("Fraudulent Transaction")
    except ValueError:
        st.write("Please ensure all input values are numbers.")