import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('AI Classifies FP Transit Exoplanets')
st.markdown('Detect false positive transiting exoplanets based on\
 their Period, Duration, Planet-Star Radius Ratio, Stellar Radius, Model train: Random Forest Classifier, Accuracy: 0.902 using Kepler database\
 \n \n ChatGPT: Machine\nlearning\nclassifiers\nhave\nbecome\nan\nincreasingly\nimportant\ntool\nin\nthe\nsearch\
 \nfor\nexoplanets.\nOne\nparticular\napplication\nis\nin\nthe\ndetection\nof\nfalse\npositive\ntransiting\nexoplanets.\
 \nFalse\npositives\ncan\narise\nwhen\nthe\nobserved\ntransit\nsignal\nis\ncaused\nby\nsomething\nother\
 \nthan\nan\norbiting\nexoplanet,\nsuch\nas\na\nbackground\neclipsing\nbinary\nstar\nor\na\nstellar\
 \nflare.\n\nMachine\nlearning\nclassifiers\nwork\nby\ntraining\nalgorithms\non\nknown\nexoplanet\ndata\
 \nand\nidentifying\npatterns\nthat\ncan\nbe\nused\nto\ndifferentiate\nbetween\nreal\nand\nfalse\npositive\ntransiting\
 \nsignals.\nThese\nalgorithms\ncan\nthen\nbe\napplied\nto\nnew\ndata\nsets\nto\ndetermine\nthe\nlikelihood\
 \nthat\na\ngiven\ntransit\nsignal\nis\ncaused\nby\nan\nexoplanet.The\nuse\nof\nmachine\nlearning\nclassifiers\
 \nhas\nbeen\nsuccessful\nin\nreducing\nthe\nnumber\nof\nfalse\npositive\ndetections\nand\nincreasing\nthe\
 \nefficiency\nof\nexoplanet\nsurveys.\nThis\nin\nturn\nhas\nhelped\nto\nimprove\nour\nunderstanding\nof\nexoplanet\
 \npopulations\nand\ntheir\ncharacteristics,\nwhich\nhas\nsignificant\nimplications\nfor\nour\nunderstanding\nof\nthe\
 \nformation\nand\nevolution\nof\nplanetary\nsystems.')

st.header("Transit Features")
col1, col2 = st.columns(2)

with col1:
    # st.text("Transit parameters")
    # sepal_l = st.slider('koi_period', 0.1, 2190.0, 0.3)
    sepal_l = st.number_input('Orbital Period [days]', step = 0.2, max_value = 2190.0, min_value = 0.3)
    sepal_w = st.number_input('Transit Duration [hrs]', step = 2.0, max_value = 138.0, min_value = 0.10)

with col2:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    petal_l = st.number_input('Planet-Star Radius Ratio', step = 0.001, max_value = 99.8, min_value = 0.001289\
        ,format="%.5f")
    petal_w = st.number_input('Stellar Radius [Solar radii]', step = 0.1, max_value = 18.0, min_value = 0.1)

st.text('')
if st.button("Predict FP / Confirmed exoplanets"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    if result[0] == "FP":
        st.text(f"{result[0]}: False posetive transit")
    else:
        st.text(f"{result[0]}: Confimed !")
