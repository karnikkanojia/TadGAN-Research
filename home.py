import streamlit as st
import pandas as pd
from orion import Orion
from utils import plot
from tensorflow.keras.utils import plot_model
import warnings

st.set_option('client.showErrorDetails', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")


@st.cache(allow_output_mutation=True)
def preprocess_data(data_file):
    data = pd.read_csv(data_file, usecols=['Date', 'No. of Trades'], parse_dates=['Date'])
    data.rename(columns={'No. of Trades': 'value', 'Date': 'timestamp'}, inplace=True)
    data.sort_values(by='timestamp', inplace=True)
    data.timestamp = (data.timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    data = data[data.timestamp > 1640975400]
    return data

@st.cache(allow_output_mutation=True)
def run_orion(data, epochs):
    known_anomalies = pd.DataFrame({
        'start': [1648751400],
        'end': [1661970599]
    })

    hyperparameters = {
        "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
            'target_column': 0,
            'window_size': 100
        },
        'keras.Sequential.LSTMSeq2Seq#1': {
            'epochs': epochs,
            'verbose': True,
            'window_size': 100,
            'input_shape': [100, 1],
            'target_shape': [100, 1]
        }
    }

    orion = Orion(
        pipeline='lstm_autoencoder',
        hyperparameters=hyperparameters
    )
    anomalies = orion.fit_detect(data)

    return anomalies, known_anomalies

def train(data_file, epochs) -> None:
    data = preprocess_data(data_file)
    anomalies, known_anomalies = run_orion(data, epochs)
    plotImg = plot(data, 'Sharpline Broadcast Ltd. (543341)', anomalies=[anomalies, known_anomalies])
    return anomalies, plotImg


st.set_page_config(layout="wide")

st.markdown("# <div style=\"text-align: center;\">Anomalies Detection</div>", unsafe_allow_html=True)
" "
" "
# Input file uploader for dataset
rawData = st.file_uploader("Add dataset")

# Epochs size slider
epochs = st.slider('Number of epochs?', 1, 15, 5)

# Button for training
__, col1, __ = st.columns([2,1,2])
if col1.button('Train🚂', use_container_width = True):

    anomalies, plotImg = train(rawData, epochs)
    __, col2, __ = st.columns([1,2,1])

    col2.dataframe(anomalies, use_container_width = True)
    st.pyplot(plotImg, True)
