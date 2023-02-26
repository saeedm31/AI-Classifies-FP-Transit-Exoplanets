import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('AI Classifies FP Transit Exoplanets')
st.markdown('Detect false positive transiting exoplanets based on\
 their Period, Duration, Planet-Star Radius Ratio, Stellar Radius, Model train: Random Forest Classifier, Accuracy ~ 0.902 (90 %) using Kepler database\
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

st.image(
            "https://bpb-us-e1.wpmucdn.com/sites.psu.edu/dist/b/10648/files/2020/09/false_positive_examples.png",
            width=600, # Manually Adjust the width of the image as per requirement
        )
st.markdown('This\nfigure\nshow\nthe\nvarious\npotential\nfalse\npositive\nsituations\nthat\ncould\nbe\nmistaken\nfor\nan\nexoplanet\nin\nphotometry.\nThese\nscenarios\ninclude\nexamples\nsuch\nas\nlow-mass\nstars\nand\nbrown\ndwarfs\nthat\nare\ntransiting\n(located\nin\nthe\nupper\nright),\na\nbackground\nmulti-star\nsystem\nthat\nis\npositioned\nclose\nto\nthe\ncandidate\nhost\nstar\n(located\nin\nthe\nlower\nleft),\nand\na\ngrazing\nstellar\nbinary\n(located\nin\nthe\nlower\nright).\nIn\norder\nto\nascertain\nwhether\na\ngiven\nsignal\nis\ntruly\ncaused\nby\na\nplanet\n(located\nin\nthe\nupper\nleft)\nor\na\nfake\nsystem,\nstatistical\nvalidation\nis\na\nuseful\ntool\n.\nThe\nimage\ncredit\nfor\nthis\nfigure\nis\nattributed\nto\nNASA\nAmes\n/\nW.\nStenzel.')

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

with col2:
   option = st.selectbox(
       "Which model would you like to use?",
       ("Model 1", "Model 2", "Model 3"),
       label_visibility=st.session_state.visibility,
       disabled=st.session_state.disabled,
   )
    
st.text('')
if st.button("Predict FP / Confirmed exoplanets"):
    result, probability = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    if result[0] == "FP":
        st.text(f"{result[0]}: False posetive transit")
        st.text(f"probability: {probability[0][1]} (0 to 1)")
    else:
        st.text(f"{result[0]}: Confimed !")
        st.text(f"probability: {probability[0][0]} (0 to 1)")
    
    if option == "Model 1":
       st.text(f"Model 1")
    else:
       st.text(f"Model 2")
