import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import base64
import matplotlib.pyplot as plt

# Set page title and icon
st.set_page_config(page_title = "Sales Buy Gender Predictions", page_icon = ":bar_chart:")

#supressing warning that comes when confusion matrix is shown
st.set_option('deprecation.showPyplotGlobalUse', False)

#Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["🏠 Home", "📂 Data Overview", "📈 EDA", "⚙️ Modeling", "🔮 Make Predictions!"])

#read in the data
df = pd.read_csv('data/final_new_df.csv')

# setting color theme
custom_theme = f"""
    <style>
        :root {{
            --primaryColor: #9C8777;
            --backgroundColor: #FFE2D3;
            --secondaryBackgroundColor: #E2C3B2;
            --textColor: #543022;
            --font: sans-serif;
        }}
    </style>
"""

st.markdown(custom_theme, unsafe_allow_html=True)

#build homepage
if page == "🏠 Home":
    st.title("💼 Company Sales By Gender Prediction")
    # Audio File 9 to 5
    st.write("🎶: Wolf Of Walstreet Audio")
    audio_file_path = "audio/it's-not-gonna-make-you-rich-and-it's-not-gonna-make-you-poor.mp3" 
    st.audio(audio_file_path, format='audio/mp3', start_time=0)
    st.subheader("This app is designed to effectively review the analytics and make predictions on company sales by gender.")
    st.write("Please use the toggle bar on the left hand side of the page to navigate between the dataset, the analytics on current employees attrition, and making predictions on future employees attrition.")
    
    # Centered image
    st.image("https://media.licdn.com/dms/image/C4D12AQHXR9nRwvEPQg/article-cover_image-shrink_600_2000/0/1520159259030?e=2147483647&v=beta&t=B8J4_PMGc8fWBm8oTKeUJt2-hpBRlQbQgEgB4QeBm_g", 
         caption="Image Caption",
         use_column_width=True,
         output_format="auto",
         width=0.5) 

#build data overview page
if page == "📂 Data Overview":
    st.title("📂 Data Overview")
    st.subheader("About the Data")
    st.write("The dataset utilized in this project comprises comprehensive information on the company's customers and sales transactions, encompassing various demographic and transactional attributes. With a total of 1000 rows and 9 columns, the dataset provides a rich source of insights into consumer behavior and purchasing patterns. It includes demographic factors such as age and gender, as well as transaction-related details such as total amount spent and product categories purchased. Through meticulous cleaning and organization, the dataset was prepared for thorough analysis, enabling the exploration of trends and patterns that inform strategic decision-making. With its diverse array of variables and substantial sample size, the dataset serves as a valuable resource for uncovering actionable insights aimed at optimizing sales strategies and enhancing customer experiences in the dynamic e-commerce landscape.")
    st.image("https://media.istockphoto.com/id/1274867768/vector/%C3%B0%C3%B1%C3%B0%C3%B0%C3%B0%C3%B0%C3%B1%C3%B0%C2%B5-rgb.jpg?s=612x612&w=0&k=20&c=uqkBqbX2Xq8TxfM-HvvYJ6jH6vUmem9H5f0gB-o7-hE=")
    st.link_button("Click here to learn more", "https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset", help = "Company Sales Dataset", type = 'primary')
    st.subheader("Quick Glance at the Data")
    # Display dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
    # Column list
    if st.checkbox("Column List"):
        st.code(f"Columns: {df.columns.tolist()}")
        if st.toggle('Further breakdown of columns'):
            num_cols = df.select_dtypes(include = 'number').columns.tolist()
            obj_cols = df.select_dtypes(include = 'object').columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    # Shape
    if st.checkbox("Shape"):
        # st.write(f"The shape is {df.shape}") -- could write it out like this or do the next instead:
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.button("Download Data as CSV"):
    # Create a link for downloading the data
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert the DataFrame to CSV, encode as base64
        href = f'<a href="data:file/csv;base64,{b64}" download="data/final_new_df.csv">Download CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)

#build EDA page
if page == "📈 EDA":
    st.title("📈 EDA")
    num_cols = df.select_dtypes(include='number').columns.tolist()
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    eda_type = st.multiselect("What type of EDA are you interested in exploring?", ["Histogram", "Box Plot", "Scatterplot"])

    # Set a custom color palette with browns and tans
    custom_palette = ['#8C564B', '#D2B48C']

    # HISTOGRAM
    if "Histogram" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for your Histogram:", num_cols, index=None)
        if h_selected_col:
            chart_title = f"Distribution of {' '.join(h_selected_col.split('_')).title()}"
            if st.toggle("Gender Hue on Histogram:"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title, color='Gender',
                                              barmode='overlay', color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title, color_discrete_sequence=custom_palette))

    # BOXPLOT
    if "Box Plot" in eda_type:
        st.subheader("Boxplots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for your Boxplot:", num_cols, index=None)
        if b_selected_col:
            chart_title = f"Distribution of {' '.join(b_selected_col.split('_')).title()}"
            if st.toggle("Gender Hue On Box Plot"):
                st.plotly_chart(px.box(df, x=b_selected_col, y='Gender', title=chart_title, color='Gender',
                                       color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title=chart_title, color_discrete_sequence=custom_palette))

    # SCATTERPLOT
    if "Scatterplot" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, index=None)
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, index=None)
        if selected_col_x and selected_col_y:
            chart_title = f"Relationship of {selected_col_x} vs {selected_col_y}"
            if st.toggle("Gender Hue On Scatterplot"):
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title, color='Gender',
                                    color_discrete_sequence=custom_palette))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title,
                                    color_discrete_sequence=custom_palette))
                
# Build Modeling Page
if page == "⚙️ Modeling":
    st.title("⚙️ Modeling")
    st.markdown("On this page, you can see how well different **machine learning** models make predictions on sales based on gender:")
    # Set up X and y
    features = ['Age', 'Product Category', 'Quantity', 'Price per Unit', 'Total Amount']
    X = df[features]
    y = df['Gender']
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # Model selection
    model_option = st.selectbox("Select a Model:", ['KNN', 'Logistic Regression', 'Random Forest'], index=None)
    if model_option:
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k):", 1, 29, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        # create a button & fit your model
        if st.button("Let's see the performance!"):
            model.fit(X_train, y_train)
            # Display results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train) * 100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test) * 100, 2)}%")
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='YlOrBr_r')
            CM_fig = plt.gcf()
            st.pyplot(CM_fig)
    if st.button("Download Model"):
        with open("trained_model.pkl", "wb") as model_file:
            pickle.dump(model, model_file)
        st.success("Model downloaded successfully!")

# Predictions Page
if page == "🔮 Make Predictions!":
    st.title("🔮 Make Predictions On Gender Based On Features For What Was Sold & When")
    st.write("This predictive model estimates the likelihood of gender based on various factors. Adjust the sliders to input information, and the model will make a prediction.")
    # Create sliders for user to input data
    age = st.slider("Age", min_value=18, max_value=64, value=30, step=1)
    product_category = st.slider("Product Category", min_value=1, max_value=3, value=2, step=1)
    quantity = st.slider("Quantity", min_value=1, max_value=4, value=2, step=1)
    price_per_unit = st.slider("Price per Unit", min_value=25, max_value=500, value=60, step=1)
    total_amount = st.slider("Total Amount", min_value=25, max_value=2000, value=2, step=1)

    # Must be in order that the model was trained on

    user_input_lr = pd.DataFrame({
        'Age': [age],
        'Product Category': [product_category],
        'Quantity': [quantity],
        'Price per Unit': [price_per_unit],
        'Total Amount': [total_amount],
    })

    features_lr = ['Age', 'Product Category', 'Quantity', 'Price per Unit', 'Total Amount']

    X_lr = df[features_lr]
    y_lr = df['Gender']

    model_lr = LogisticRegression()

    if st.button("Make a Prediction! (Gender: 1 = Male | Attrition: 2 = Female)"):
        model_lr.fit(X_lr, y_lr)  # Fit the model here
        prediction_lr = model_lr.predict(user_input_lr)
        st.write(f"{model_lr} predicts the gender as {prediction_lr[0]}.")
        st.balloons()
        if prediction_lr[0] == 0:
            st.subheader("Gender 1 = Male!")
        else:
            st.subheader("Gender 2 = Female")
    
    prediction_proba_lr = model_lr.predict_proba(user_input_lr)[:, 1]
    st.write(f"The predicted gender is: {prediction_proba_lr[0]:.2%}")