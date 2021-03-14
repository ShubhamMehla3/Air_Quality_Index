import streamlit as st
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime
from datetime import date, timedelta

import pickle
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go


#sktime models
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
#from sktime.utils.plotting.forecasting import plot_ys
#from sktime.forecasting.arima import AutoARIMA

import warnings
warnings.filterwarnings('ignore')

st.title('Air Quality Index')

@st.cache
def tsplot(y, lags=None, figsize=(20, 12), style='bmh'):
    """
        Plot time series, its ACF and PACF
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

def null_values(df):
    null_test = (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False).index
    null_data_test = pd.concat([
        df.isnull().sum(axis = 0),
        (df.isnull().sum(axis = 0)/len(df)).sort_values(ascending=False),
        df.loc[:, df.columns.isin(list(null_test))].dtypes], axis=1)
    null_data_test = null_data_test.rename(columns={0: '# null', 
                                        1: '% null', 
                                        2: 'type'}).sort_values(ascending=False, by = '% null')
    null_data_test = null_data_test[null_data_test["# null"]!=0]
    
    return null_data_test

def types(df):
    return pd.DataFrame(df.dtypes, columns=['Type'])
#
# def forecasting_autoarima(y_train, y_test):
#     fh = np.arange(len(y_test)) + 1
#     forecaster = AutoARIMA()
#     forecaster.fit(y_train)
#     y_pred = forecaster.predict(fh)
#     plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
#     st.pyplot()

# def forecasting_fbprophet(new_df):
#
#     new_df.rename(columns={'date':'ds','aqi':'y'})
#
#     m = Prophet()
#     m.fit(new_df)
#     future = m.make_future_dataframe(periods=periods_input)
#
#     forecast = m.predict(future)
#     fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#
#     fcst_filtered = fcst[fcst['ds'] > max_date]
#     st.write(fcst_filtered)
#
#     """
#     The next visual shows the actual (black dots) and predicted (blue line) values over time.
#     """
#     fig1 = m.plot(forecast)
#     st.write(fig1)
#
#     """
#     The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
#     """
#     fig2 = m.plot_components(forecast)
#     st.write(fig2)
#     """
#     ### Download the Forecast Data
#
#     The below link allows you to download the newly created forecast to your computer for further analysis and use.
#     """
#     csv_exp = fcst_filtered.to_csv(index=False)
#     # When no file name is given, pandas returns the CSV as a string, nice.
#     b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
#     href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
#     st.markdown(href, unsafe_allow_html=True)


def predict_fbprophet(start, end):
    pickle_in = open('final.pkl', 'rb')
    model = pickle.load(pickle_in)
    cols = ['ds']
    value = (start, end)
    values = list(value)
    date_obj = pd.to_datetime(values)
    final = np.array(date_obj)
    data_unseen = pd.DataFrame(values, columns=cols)
    forecast = model.predict(data_unseen)
    prediction = forecast['yhat']
    st.success('AQI VALUE IS  {}'.format(round(prediction[1], 3)))
    result =round(prediction[1], 3)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Speed", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'green'},
                {'range': [51, 100], 'color': 'yellow'},
                {'range': [101, 150], 'color': 'orange'},
                {'range': [151, 200], 'color': 'red'},
                {'range': [201, 300], 'color': 'purple'},
                {'range': [301, 500], 'color': 'maroon'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 500}}))

    fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})

    st.write(fig)
    delta = end - start  # as timedelta

    future = model.make_future_dataframe(periods=delta.days)
    new = model.predict(future)
    """
        The next visual shows the actual (black dots) and predicted (blue line) values over time.
        """
    fig1 = model.plot(new)

    st.write(fig1)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = model.plot_components(new)
    st.write(fig2)
    return round(prediction[1], 3)
    #return round(prediction[1], 3)


def main():
    # Upload file
    type_of_input = st.selectbox("Select One ", ["upload file", "Manually(giving the inputs)"])

    if type_of_input=="upload file":

        st.sidebar.title("What to do")
        activities = ["Exploratory Data Analysis", "Data Visualization", "Model Building", "About"]
        choice = st.sidebar.selectbox("Select Activity", activities)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None and choice == "Exploratory Data Analysis":
            data = pd.read_csv(uploaded_file)
            st.subheader(choice)
            # Show dataset
            if st.checkbox("Show Dataset"):
                rows = st.number_input("Number of rows", 5, len(data))
                st.dataframe(data.head(rows))
            # Show columns
            if st.checkbox("Columns"):
                st.write(data.columns)
            # Data types
            if st.checkbox("Column types"):
                st.write(types(data))
            # Show Shape
            if st.checkbox("Shape of Dataset"):
                data_dim = st.radio("Show by", ("Rows", "Columns", "Shape"))
                if data_dim == "Columns":
                    st.text("Number of Columns: ")
                    st.write(data.shape[1])
                elif data_dim == "Rows":
                    st.text("Number of Rows: ")
                    st.write(data.shape[0])
                else:
                    st.write(data.shape)
            # Check null values in dataset
            if st.checkbox("Check null values"):
                nvalues = null_values(data)
                st.write(nvalues)
            # Show Data summary
            if st.checkbox("Show Data Summary"):
                st.text("Datatypes Summary")
                st.write(data.describe())
            # Plot time series, ACF and PACF
            if st.checkbox("Select column as time series"):
                columns = data.columns.tolist()
                selected = st.multiselect("Choose", columns)
                series = data[selected]

        elif uploaded_file is not None and choice == "Data Visualization":
            st.subheader(choice)
            data = pd.read_csv(uploaded_file)
            df = data.copy()
            all_columns = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["area", "line", "scatter", "bar", "correlation", "distribution"])

            if type_of_plot=="line":
                select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
                cust_data = df[select_columns_to_plot]
                st.line_chart(cust_data)

            elif type_of_plot=="area":
                select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
                cust_data = df[select_columns_to_plot]
                st.area_chart(cust_data)

            elif type_of_plot=="bar":
                select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
                cust_data = df[select_columns_to_plot]
                st.bar_chart(cust_data)


            elif type_of_plot=="correlation":
                st.write(sns.heatmap(df.corr(), annot=True, linewidths=.5, annot_kws={"size": 7}))
                st.pyplot()

            elif type_of_plot=="scatter":
                st.write("Scatter Plot")
                scatter_x = st.selectbox("Select a column for X Axis", all_columns)
                scatter_y = st.selectbox("Select a column for Y Axis", all_columns)
                st.write(sns.scatterplot(x=scatter_x, y=scatter_y, data = df))
                st.pyplot()

            elif type_of_plot=="distribution":
                select_columns_to_plot = st.multiselect("Select columns to plot", all_columns)
                st.write(sns.distplot(df[select_columns_to_plot]))
                st.pyplot()

        elif uploaded_file is not None and choice == "Model Building":
            st.subheader(choice)
            data = pd.read_csv(uploaded_file)
            df = data.copy()
            st.write("Select the columns to use for training")
            columns = df.columns.tolist()
            selected_column = st.multiselect("Select Columns", columns)
            new_df = df[selected_column]
            st.write(new_df)

            if (st.button("Training a Model")):
                model_selection = st.selectbox("Model to train", ["AutoArima" , "LSTM", "MLP", "RNN"])

                # if model_selection == "FBprophet":
                #     forecasting_fbprophet(new_df)
                # if model_selection == "AutoArima":
                #     y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])
                #     forecasting_autoarima(y_train, y_test)



        elif choice == "About":
            st.title("About")
            st.write("The app developed by ML-IOT TEAM.")
            st.write("Stack: Python, Streamlit, Docker, Kubernetes")
    elif type_of_input=="Manually(giving the inputs)":
        st.sidebar.title("What To Do")
        activities = ["AQI  value Prediction ", "About"]
        choice = st.sidebar.selectbox('Select Here', activities)
        if choice=="AQI  value Prediction ":
            start = st.date_input('Start date')
            end = st.date_input('End date')
            model_selection = st.selectbox("Model to train", ["FBProphet", "LSTM", "MLP", "RNN"])
            if model_selection == "FBProphet":
                if st.button("Predict"):
                    result = predict_fbprophet(start, end)

        elif choice == "About":
            st.title("About")
            st.write("The app developed by ML-IOT TEAM.")
            st.write("Stack: Python, Streamlit, Docker, Kubernetes")

            #y_train, y_test = temporal_train_test_split(new_df.T.iloc[0])


if __name__ == "__main__":
    main()