import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("<h1 style='text-align: center; font-size: 6em;'>Diamond Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 2em;'>TESTING MACHINE LEARNING MODEL</h1>", unsafe_allow_html=True)
st.markdown('---')

df = pd.read_excel('data/Diamond_price(cleaned).xlsx')
RandomForestRegressor = joblib.load('model/RFRegressor.pkl')


def predict_price(carat, cut, color, clarity, total_depth, table, length, width, depth):
    RFR_prediction = RandomForestRegressor.predict([[carat, cut, color, clarity, total_depth, table, length, width, depth]])
    return RFR_prediction


def main():
    dict_cut = {'Poor': 5, 'Fair': 4, 'Good': 3, 'Very Good': 2, 'Excellent': 1}
    dict_color = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
    dict_clarity = {'I1': 8, 'SI2': 7, 'SI1': 6, 'VS2': 5, 'VS1': 4, 'VVS2': 3, 'VVS1': 2, 'IF': 1}

    # Input components
    st.image('images/hand_diamond_round_uk.png', caption='Diamond Size', use_column_width='always')
    carat = st.slider('*Carat*', min_value=0.2, max_value=5.0, step=0.1, value=1.0)
    st.markdown('---')
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.image('images/brilliance-diamond-cut-chart.webp', caption='Cutting Standard')
    st.write("Cut (Quality of the cut):")
    cut_key = st.selectbox('*Cut*', ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    st.markdown('---')
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.image('images/color-grade-chart.jpg', caption='Color Standard')
    st.write("Color (Diamond color, from J (worst) to D (best)):")
    color_key = st.selectbox('*Color*', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    st.markdown('---')
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.image('images/diamond-clarity-scale.jpg', caption='Clarity Standard')
    st.write("Clarity (A measurement of how clear the diamond is):")
    clarity_key = st.selectbox('*Clarity*', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    st.markdown('---')
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.image('images/Anatomy-of-a-diamond.jpg', caption='Diamond Dimension', use_column_width='always')
    col1, col3 = st.columns(2)
    with col1:
        st.write("Total depth percentage = Depth / mean(Length, Width) = 2 * Depth / (Length + Width):")
        total_depth = st.number_input('*Total Depth %*', min_value=0.0, max_value=100.0, value=61.74)

        st.write("Table (Width of top of diamond relative to widest point):")
        table = st.number_input('*Table Size*', min_value=0.0, max_value=100.0, value=57.45)

    with col3:
        st.write("Length (Length in mm):")
        length = st.number_input('*Length*', min_value=3.73, max_value=50.0, value=5.73)

        st.write("Width (Width in mm):")
        width = st.number_input('*Width*', min_value=3.68, max_value=50.0, value=5.73)

        st.write("Depth (Depth in mm):")
        depth = st.number_input('*Depth*', min_value=1.07, max_value=50.0, value=3.53)

    cut = dict_cut[cut_key]
    color = dict_color[color_key]
    clarity = dict_clarity[clarity_key]

    # Button to trigger prediction
    if st.button('Predict', use_container_width=True):
        RFR = predict_price(carat, cut, color, clarity, total_depth, table, length, width, depth)
        st.success(f'Predicted Price: ${RFR[0]:,.2f}')

    st.subheader("Correlation Map")
    corr = df.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
    st.pyplot(fig)


if __name__ == '__main__':
    main()

if st.checkbox('Show Model Data'):
    st.dataframe(df)



