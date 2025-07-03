import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="AI Classifies FP Transit Exoplanets",
    page_icon="🌌",
    layout="wide"
)

# Function to train and predict
@st.cache_resource
def train_model():
    """Train the model (cached to avoid retraining)"""
    try:
        # Load and prepare data
        df = pd.read_csv("FP_new.csv")
        X = df[["koi_period", "koi_duration", "koi_ror", "koi_srad"]]
        y = df["info_status"]
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

def predict_exoplanet(data):
    """Make prediction using the trained model"""
    model = train_model()
    if model is not None:
        return model.predict(data), model.predict_proba(data)
    else:
        return None, None

# Sidebar for navigation
st.sidebar.title("Navigation")

# Use buttons instead of dropdown
if st.sidebar.button("🏠 Prediction", use_container_width=True):
    st.session_state.page = "Prediction"

if st.sidebar.button("📊 Data Analysis", use_container_width=True):
    st.session_state.page = "Data Analysis"

# Initialize page if not set
if 'page' not in st.session_state:
    st.session_state.page = "Prediction"

page = st.session_state.page

if page == "Prediction":
    # Original prediction page
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
    col1, col2, col3 = st.columns(3)

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

    with col3:
        option = st.selectbox("Which model would you like to use?",("Model 1", "Model 2", "Model 3"))
        
    st.text('')
    if st.button("Predict FP / Confirmed exoplanets"):
        result, probability = predict_exoplanet(
            np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
        if result is not None and probability is not None:
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
        else:
            st.error("Failed to load model. Please check if FP_new.csv exists.")

elif page == "Data Analysis":
    # New data analysis page
    st.title("Data Analysis - Feature Distributions")
    st.markdown("This page shows the distribution of the four features used in training the model:")
    st.markdown("- **koi_period**: Orbital Period [days]")
    st.markdown("- **koi_duration**: Transit Duration [hrs]") 
    st.markdown("- **koi_ror**: Planet-Star Radius Ratio")
    st.markdown("- **koi_srad**: Stellar Radius [Solar radii]")
    
    # Load the data
    try:
        df = pd.read_csv("FP_new.csv")
        
        # Create a figure with subplots for each feature
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution of Features Used in Training', fontsize=16, fontweight='bold')
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Feature names and their display names
        features = {
            'koi_period': 'Orbital Period [days]',
            'koi_duration': 'Transit Duration [hrs]', 
            'koi_ror': 'Planet-Star Radius Ratio',
            'koi_srad': 'Stellar Radius [Solar radii]'
        }
        
        # Create distribution plots
        for idx, (feature, title) in enumerate(features.items()):
            row = idx // 2
            col = idx % 2
            
            # Create histogram with KDE
            sns.histplot(data=df, x=feature, hue='info_status', 
                        bins=30, alpha=0.7, kde=True, ax=axes[row, col])
            
            axes[row, col].set_title(f'Distribution of {title}', fontweight='bold')
            axes[row, col].set_xlabel(title)
            axes[row, col].set_ylabel('Count')
            axes[row, col].legend(title='Status', labels=['Confirmed', 'False Positive'])
            
            # Add statistics
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            axes[row, col].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                               transform=axes[row, col].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        summary_stats = df[['koi_period', 'koi_duration', 'koi_ror', 'koi_srad']].describe()
        st.dataframe(summary_stats)
        
        # Display class distribution
        st.subheader("Class Distribution")
        class_counts = df['info_status'].value_counts()
        st.write(class_counts)
        
        # Create a pie chart for class distribution
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Confirmed vs False Positive Exoplanets')
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure the 'FP_new.csv' file is in the same directory as the app.")    