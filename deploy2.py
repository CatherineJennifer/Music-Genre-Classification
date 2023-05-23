import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Function to load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset='filename', inplace=True)
    df = df.drop(labels='filename', axis=1)
    class_list = df.iloc[:, -1]
    convertor = LabelEncoder()
    y = convertor.fit_transform(class_list)
    X = StandardScaler().fit_transform(np.array(df.iloc[:, :-1], dtype=float))
    return X, y, convertor, df.columns[:-1]

# Function to train the model
def train_model(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    return model, x_test, y_test

# Function to display model performance metrics
def display_performance(model, x_test, y_test, convertor):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write("Accuracy: ", accuracy)

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=convertor.classes_, yticklabels=convertor.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.write(fig)

    report = classification_report(y_test, predictions, target_names=convertor.classes_)
    st.code(report)

# Function to visualize charts for all columns
def visualize_charts(data):
    st.header("Data Visualization")
    for column in data.columns:
        fig = px.histogram(data, x=column, title=column)
        st.plotly_chart(fig)

# Streamlit app
def main():
    st.title("Music Genre Classification")

    # File upload and data preprocessing
    st.header("Data Selection")
    file_path = st.file_uploader("Upload a features CSV file", type=["csv"])
    if file_path is not None:
        X, y, convertor, columns = load_data(file_path)
        model, x_test, y_test = train_model(X, y)

        # Model performance
        st.header("Model Performance")
        display_performance(model, x_test, y_test, convertor)

        # Data visualization
        visualize_charts(pd.DataFrame(X, columns=columns))

if __name__ == '__main__':
    main()
