import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('Nifty50_model.pkl', 'rb'))
def main():
    st.title('Nifty50 Price Prediction')
    #input variable
    Open = st.number_input('Enter open price of stock')
    High = st.number_input("Enter high price of stock")
    Low = st.number_input("Enter low price of stock")

    if st.button("Predict close price of stock"):
        df = pd.DataFrame({'Open': [Open], 'High': [High], 'Low': [Low]})
        sc = StandardScaler()
        df = sc.fit_transform(df)
        output = model.predict(df)
        st.header('Your stock close price according to our model is {}'.format(output))

if __name__=='__main__':
    main()