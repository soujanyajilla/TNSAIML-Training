import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

df = pd.read_csv("dataset/retail_sales_forecast_dataset.csv")
model = joblib.load("model.joblib")

st.title(" 🛒Retail Sales Forecasting")

st.subheader("Sales Overview")
col1, col2 = st.columns(2)
col1.metric("Total Units Sold", int(df["Units_Sold"].sum()))
col2.metric("Average Sales", round(df["Units_Sold"].mean(), 2))

st.sidebar.header("Sales Information")

Year = st.sidebar.selectbox(
    "Enter Year", sorted(df["Year"].dropna().unique())
)

Month = st.sidebar.selectbox("Enter Month", list(range(1, 13)))

Product_Category = st.sidebar.selectbox(
    "Product Category", sorted(df["Product_Category"].dropna().unique())
)

Region = st.sidebar.selectbox(
    "Region", sorted(df["Region"].dropna().unique())
)

Weather_Condition = st.sidebar.selectbox(
    "Weather Condition", sorted(df["Weather_Condition"].dropna().unique())
)

Season = st.sidebar.selectbox(
    "Season", sorted(df["Season"].dropna().unique())
)

Inventory_Level = st.sidebar.number_input(
    "Inventory Level", min_value=0, value=100
)

Unit_Price = st.sidebar.number_input(
    "Unit Price", min_value=0.0, value=50.0
)

Discount_Percent = st.sidebar.number_input(
    "Discount Percentage", min_value=0.0, max_value=100.0, value=0.0
)

Promotion = st.sidebar.selectbox(
    "Promotion", sorted(df["Promotion"].dropna().unique())
)

Competitor_Price = st.sidebar.number_input(
    "Competitor Price", min_value=0.0, value=50.0
)


input_values = pd.DataFrame([{
    "Year": Year,
    "Month": Month,
    "Product_Category": Product_Category,
    "Region": Region,
    "Weather_Condition": Weather_Condition,
    "Season": Season,
    "Inventory_Level": Inventory_Level,
    "Unit_Price": Unit_Price,
    "Discount_Percent": Discount_Percent,
    "Promotion": Promotion,
    "Competitor_Price": Competitor_Price
}])


st.subheader("Details of the Sales Information")
st.dataframe(input_values, use_container_width=True)


if st.button("Predict Sales", type="primary"):
    prediction = model.predict(input_values)[0]
    prediction = max(0, round(float(prediction)))

    st.subheader("Prediction Result")
    st.success(f"Predicted Units Sold: {prediction} products")


st.subheader("📊Historical Sales Trends")

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    sales = df.dropna(subset=["Date"]).set_index("Date")
    st.line_chart(sales["Units_Sold"].resample("ME").sum())
