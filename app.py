
import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Local Food Wastage Management", layout="wide")

DB_PATH = "food_wastage.db"

# --- DB helpers ---
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def run_query(sql: str, params: tuple = ()):
    conn = get_conn()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception as e:
        st.warning(f"Query failed: {e}")
        return pd.DataFrame()

# --- Sidebar filters (loaded defensively) ---
st.sidebar.title("Filters")

def get_distinct_values(table, column):
    try:
        df = run_query(f"SELECT DISTINCT {column} AS v FROM {table} WHERE {column} IS NOT NULL ORDER BY 1;")
        return ["All"] + df["v"].dropna().astype(str).tolist()
    except Exception:
        return ["All"]

city_options = sorted(list(set(
    [*get_distinct_values("providers", "City"), *get_distinct_values("food_listings", "Location")]
)))
provider_type_options = get_distinct_values("providers", "Type")
food_type_options = get_distinct_values("food_listings", "Food_Type")
meal_type_options = get_distinct_values("food_listings", "Meal_Type")
status_options = get_distinct_values("claims", "Status")

city_filter = st.sidebar.selectbox("City / Location", city_options)
provider_type_filter = st.sidebar.selectbox("Provider Type", provider_type_options)
food_type_filter = st.sidebar.selectbox("Food Type", food_type_options)
meal_type_filter = st.sidebar.selectbox("Meal Type", meal_type_options)
status_filter = st.sidebar.selectbox("Claim Status", status_options)

# Helper to build WHERE clauses
def add_where(clauses):
    clauses = [c for c in clauses if c]
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""

# --- KPI summary ---
st.title("🍽️ Local Food Wastage Management Dashboard")

kpi1 = run_query("SELECT COUNT(*) AS c FROM providers")
kpi2 = run_query("SELECT COUNT(*) AS c FROM receivers")
kpi3 = run_query("SELECT COUNT(*) AS c FROM food_listings")
kpi4 = run_query("SELECT COUNT(*) AS c FROM claims")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Providers", int(kpi1["c"].iloc[0]) if not kpi1.empty else 0)
col2.metric("Receivers", int(kpi2["c"].iloc[0]) if not kpi2.empty else 0)
col3.metric("Food Listings", int(kpi3["c"].iloc[0]) if not kpi3.empty else 0)
col4.metric("Claims", int(kpi4["c"].iloc[0]) if not kpi4.empty else 0)

st.markdown("---")

# --- Query 1: Providers per city (Top 10) ---
st.subheader("Top Cities by Number of Providers")
where = add_where([
    None if city_filter == "All" else "City = ?",
    None if provider_type_filter == "All" else "Type = ?"
])
params = tuple([p for p, cond in [
    (city_filter, city_filter != "All"),
    (provider_type_filter, provider_type_filter != "All")
] if cond])

query1 = f"""
SELECT City, COUNT(*) AS provider_count
FROM providers
{where}
GROUP BY City
ORDER BY provider_count DESC
LIMIT 10;
"""
df1 = run_query(query1, params)
st.dataframe(df1, use_container_width=True)
if not df1.empty:
    fig, ax = plt.subplots()
    ax.bar(df1["City"].astype(str), df1["provider_count"])
    ax.set_xlabel("City")
    ax.set_ylabel("Providers")
    ax.set_title("Top Cities by Providers")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --- Query 2: Provider type contributions ---
st.subheader("Provider Type Contributions")
where = add_where([
    None if city_filter == "All" else "City = ?"
])
params = tuple([p for p, cond in [
    (city_filter, city_filter != "All")
] if cond])

query2 = f"""
SELECT Type, COUNT(*) AS count
FROM providers
{where}
GROUP BY Type
ORDER BY count DESC;
"""
df2 = run_query(query2, params)
st.dataframe(df2, use_container_width=True)
if not df2.empty:
    fig, ax = plt.subplots()
    ax.bar(df2["Type"].astype(str), df2["count"])
    ax.set_xlabel("Provider Type")
    ax.set_ylabel("Count")
    ax.set_title("Provider Type Contributions")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --- Query 7: Most common food types ---
st.subheader("Most Common Food Types")
where = add_where([
    None if city_filter == "All" else "Location = ?",
    None if meal_type_filter == "All" else "Meal_Type = ?"
])
params = tuple([p for p, cond in [
    (city_filter, city_filter != "All"),
    (meal_type_filter, meal_type_filter != "All")
] if cond])

query7 = f"""
SELECT Food_Type, COUNT(*) AS type_count
FROM food_listings
{where}
GROUP BY Food_Type
ORDER BY type_count DESC;
"""
df7 = run_query(query7, params)
st.dataframe(df7, use_container_width=True)
if not df7.empty:
    fig, ax = plt.subplots()
    ax.bar(df7["Food_Type"].astype(str), df7["type_count"])
    ax.set_xlabel("Food Type")
    ax.set_ylabel("Count")
    ax.set_title("Most Common Food Types")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --- Query 10: % Claims by status ---
st.subheader("Percentage of Claims by Status")
where = add_where([
    None if status_filter == "All" else "Status = ?"
])
params = tuple([p for p, cond in [
    (status_filter, status_filter != "All")
] if cond])

query10_total = f"SELECT COUNT(*) AS total FROM claims{where};"
total_df = run_query(query10_total, params)
total = int(total_df["total"].iloc[0]) if not total_df.empty else 0

query10 = f"""
SELECT Status,
       COUNT(*) * 100.0 / {max(total,1)} AS percentage
FROM claims
{where}
GROUP BY Status;
"""
df10 = run_query(query10, params)
st.dataframe(df10, use_container_width=True)
if not df10.empty:
    fig, ax = plt.subplots()
    ax.bar(df10["Status"].astype(str), df10["percentage"])
    ax.set_xlabel("Status")
    ax.set_ylabel("Percentage")
    ax.set_title("Claims by Status (%)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Query 12: Most claimed meal type ---
st.subheader("Most Claimed Meal Type")
where = add_where([
    None if meal_type_filter == "All" else "f.Meal_Type = ?"
])
params = tuple([p for p, cond in [
    (meal_type_filter, meal_type_filter != "All")
] if cond])

query12 = f"""
SELECT f.Meal_Type, COUNT(c.Claim_ID) AS claims_count
FROM food_listings f
JOIN claims c ON f.Food_ID = c.Food_ID
{where}
GROUP BY f.Meal_Type
ORDER BY claims_count DESC;
"""
df12 = run_query(query12, params)
st.dataframe(df12, use_container_width=True)
if not df12.empty:
    fig, ax = plt.subplots()
    ax.bar(df12["Meal_Type"].astype(str), df12["claims_count"])
    ax.set_xlabel("Meal Type")
    ax.set_ylabel("Claims")
    ax.set_title("Most Claimed Meal Type")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Query 14: Top locations by completed claims ---
st.subheader("Top 5 Locations by Completed Claims")
where = add_where([
    None if city_filter == "All" else "f.Location = ?"
])
params = tuple([p for p, cond in [
    (city_filter, city_filter != "All")
] if cond])

query14 = f"""
SELECT f.Location, COUNT(c.Claim_ID) AS completed_claims
FROM food_listings f
JOIN claims c ON f.Food_ID = c.Food_ID
WHERE c.Status = 'Completed'{" AND " + where[7:] if where else ""}
GROUP BY f.Location
ORDER BY completed_claims DESC
LIMIT 5;
"""
df14 = run_query(query14, params)
st.dataframe(df14, use_container_width=True)
if not df14.empty:
    fig, ax = plt.subplots()
    ax.bar(df14["Location"].astype(str), df14["completed_claims"])
    ax.set_xlabel("Location")
    ax.set_ylabel("Completed Claims")
    ax.set_title("Top 5 Locations by Completed Claims")
    plt.xticks(rotation=0)
    st.pyplot(fig)

# --- Extra analyses (tables only) ---
st.subheader("Receivers with Most Claims")
query4 = """
SELECT r.Name, COUNT(c.Claim_ID) AS claims_count
FROM receivers r
JOIN claims c ON r.Receiver_ID = c.Receiver_ID
GROUP BY r.Name
ORDER BY claims_count DESC
LIMIT 10;
"""
st.dataframe(run_query(query4), use_container_width=True)

st.subheader("Claims per Food Item")
query8 = """
SELECT f.Food_Name, COUNT(c.Claim_ID) AS claims_count
FROM food_listings f
JOIN claims c ON f.Food_ID = c.Food_ID
GROUP BY f.Food_Name
ORDER BY claims_count DESC
LIMIT 15;
"""
st.dataframe(run_query(query8), use_container_width=True)

st.subheader("Total Quantity Donated by Provider")
query13 = """
SELECT p.Name, SUM(f.Quantity) AS total_donated
FROM providers p
JOIN food_listings f ON p.Provider_ID = f.Provider_ID
GROUP BY p.Name
ORDER BY total_donated DESC
LIMIT 15;
"""
st.dataframe(run_query(query13), use_container_width=True)

st.caption("Tip: Use the filters in the sidebar to refine the dashboard.")
