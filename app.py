from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import numpy as np
import streamlit as st
import pandas as pd
import os
from io import BytesIO
from openai import OpenAI
from datetime import datetime
import json

import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import tempfile

import json
import os

import json
from datetime import datetime, timedelta
import random

LICENSE_FILE = "licenses.json"


def load_licenses():
    try:
        with open(LICENSE_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_licenses(licenses):
    with open(LICENSE_FILE, "w") as f:
        json.dump(licenses, f, indent=2)


def activate_license(otp_input):
    licenses = load_licenses()
    license = licenses.get(otp_input)
    if not license:
        return None, "❌ Invalid code"
    if license["activated"]:
        return None, "🔒 Code has already been used"
    if datetime.now() > datetime.fromisoformat(license["expires"]):
        return None, "⌛ Code expired"
    license["activated"] = True
    licenses = load_licenses()
    licenses[otp_input] = license
    save_licenses(licenses)

    return license, "✅ Code activated"


# Initialize OpenAI client
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(page_title="Data Analyzer", layout="wide")
st.title("📊 Data Analyzer - AI Enhanced")

# 🔐 Secure Access
st.sidebar.header("🔐 Secure Access")

# Initialize session if not already
if "otp_activated" not in st.session_state:
    try:
        with open("session.json", "r") as f:
            session_data = json.load(f)
        expiry = datetime.fromisoformat(session_data["expires"])
        if datetime.now() < expiry:
            st.session_state.otp_activated = True
        else:
            st.session_state.otp_activated = False
    except:
        st.session_state.otp_activated = False


if not st.session_state.otp_activated:
    otp_input = st.sidebar.text_input(
        "Enter One-Time Access Code", type="password")

    if otp_input:
        license, msg = activate_license(otp_input)
        st.sidebar.info(msg)

        if license:
            # ✅ Mark OTP as used
            license["activated"] = True
            save_licenses(load_licenses())

            # ✅ Save session (30 days from now)
            session_data = {
                "expires": (datetime.now() + timedelta(days=30)).isoformat()
            }
            with open("session.json", "w") as f:
                json.dump(session_data, f)

            st.session_state.otp_activated = True
        else:
            st.stop()
    else:
        st.stop()


# 🎨 Theme Switcher (always visible)
theme = st.sidebar.selectbox(
    "Choose Theme",
    options=["Light", "Dark"],
    index=0
)

if theme == "Dark":
    bg_color = "#1e1e1e"
    text_color = "#f1f1f1"
    accent_color = "#0f62fe"
else:
    bg_color = "#ffffff"
    text_color = "#000000"
    accent_color = "#0f62fe"

# Inject Theme Styles
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stButton>button {{
            color: white;
            background-color: {accent_color};
        }}
        .stSelectbox > div > div {{
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


def smart_merge(files):
    dataframes = []
    for file in files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        dataframes.append(df)

    # Smart merge: find common columns
    if len(dataframes) > 1:
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols &= set(df.columns)
        if common_cols:
            merged_df = pd.concat([df[list(common_cols)]
                                   for df in dataframes], ignore_index=True)
        else:
            # Fallback: merge all by column order
            merged_df = pd.concat(dataframes, ignore_index=True)
    elif dataframes:
        merged_df = dataframes[0]
    else:
        merged_df = pd.DataFrame()

    return merged_df


# ✅ Upload Files
uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)


df = None

if uploaded_files:
    try:
        df = smart_merge(uploaded_files)
        if df.empty:
            st.error("❌ No valid data found in uploaded files.")
        else:
            st.success("✅ Files merged successfully.")
            st.subheader("📄 Merged Raw Data")
            st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Error reading files: {e}")


st.sidebar.markdown("### 🔐 Premium PDF Customization")
enable_custom_pdf = st.sidebar.checkbox("Enable Custom PDF")


if enable_custom_pdf:
    custom_title = st.sidebar.text_input(
        "Report Title", value="Cleaned Data Summary Report")
    client_name = st.sidebar.text_input("Client Name")
    footer_text = st.sidebar.text_input(
        "Footer Text", value="Yahnova Automations")
    logo_file = st.sidebar.file_uploader("Upload Logo", type=["png"])
else:
    custom_title = "Cleaned Data Summary Report"
    client_name = ""
    footer_text = "Yahnova Automations"
    logo_file = None


if enable_custom_pdf:
    st.sidebar.markdown("### 🎨 PDF Theme")

    selected_font = st.sidebar.selectbox(
        "Font", ["Arial", "Courier", "Helvetica", "Times"])
    selected_color = st.sidebar.color_picker(
        "Header & Chart Title Color", "#003366")
else:
    selected_font = "Arial"
    selected_color = "#003366"


@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


def auto_detect_columns(df):
    column_tags = {}

    for col in df.columns:
        col_lower = col.lower()

        if "date" in col_lower:
            column_tags[col] = "📅 Date"
        elif any(word in col_lower for word in ["amount", "amt", "value", "total", "debit", "credit"]):
            column_tags[col] = "💰 Amount"
        elif any(word in col_lower for word in ["description", "details", "narration", "note"]):
            column_tags[col] = "📝 Description"
        elif any(word in col_lower for word in ["name", "account", "client", "customer"]):
            column_tags[col] = "👤 Account Name"
        elif any(word in col_lower for word in ["type", "category", "nature", "transaction"]):
            column_tags[col] = "🔄 Transaction Type"
        else:
            column_tags[col] = "📦 Other"

    return column_tags


def detect_anomalies(df):
    anomalies = pd.DataFrame()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std == 0 or pd.isna(col_std):  # Avoid divide-by-zero
            continue

        z_scores = (df[col] - col_mean) / col_std
        anomaly_mask = (z_scores.abs() > 3)  # Z-score threshold

        if anomaly_mask.any():
            temp = df[anomaly_mask].copy()
            temp["Anomaly Column"] = col
            temp["Z-Score"] = z_scores[anomaly_mask]
            anomalies = pd.concat([anomalies, temp], axis=0)

    return anomalies.reset_index(drop=True)


def generate_pdf_summary(df, custom_title, footer_text, client_name, logo_file, font_family, color_hex):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from fpdf import FPDF
    from tempfile import NamedTemporaryFile, mkdtemp
    from datetime import datetime

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    stats = df.describe(include='all').fillna("").round(2)
    nulls = df.isnull().sum().sum()

    tmp_dir = mkdtemp()
    chart_paths = []

    # Chart 1 & 2: Histograms
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) >= 1:
        for col in numeric_cols[:2]:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, color='skyblue')
            plt.title(f'Distribution of {col}')
            chart_path = os.path.join(tmp_dir, f"{col}_hist.png")
            plt.savefig(chart_path)
            chart_paths.append((chart_path, f"Distribution of {col}"))
            plt.close()

    # Chart 3: Correlation Heatmap
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(6, 5))
        corr = df[numeric_cols].corr().round(2)
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(tmp_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        chart_paths.append((heatmap_path, "Correlation Heatmap"))
        plt.close()

    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(150)
            date_str = datetime.now().strftime("%B %d, %Y")
            self.cell(
                0, 10, f"{footer_text} | {date_str} | Page {self.page_no()}", 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()

    # Logo insertion
    if logo_file is not None:
        tmp_logo_path = os.path.join(tmp_dir, "custom_logo.png")
        with open(tmp_logo_path, "wb") as f:
            f.write(logo_file.read())
        logo_path = tmp_logo_path
    else:
        logo_path = "logo.png"

    if os.path.exists(logo_path):
        logo_width_mm = 50
        logo_height_mm = 20
        page_width_mm = 210
        x_centered = (page_width_mm - logo_width_mm) / 2
        pdf.image(logo_path, x=x_centered, y=8,
                  w=logo_width_mm, h=logo_height_mm)
        pdf.ln(logo_height_mm + 15)
    else:
        pdf.ln(10)

    # Title and optional client name
# Title and optional client name
    pdf.set_font(font_family, "B", 14)
    pdf.set_text_color(*hex_to_rgb(color_hex))  # Apply user color
    pdf.cell(200, 10, txt=custom_title, ln=True, align="C")
    pdf.set_text_color(0, 0, 0)  # Reset to black

    if client_name:
        pdf.set_font(font_family, "I", 12)
        pdf.cell(200, 10, txt=f"Client: {client_name}", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font(font_family, size=13)
    pdf.cell(200, 10, txt=f"Total Rows: {df.shape[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Total Columns: {df.shape[1]}", ln=True)
    pdf.cell(200, 10, txt=f"Missing Values Filled: {nulls}", ln=True)
    pdf.ln(5)

    pdf.set_font(font_family, size=10)
    for col in stats.columns:
        values = ', '.join(
            [f"{k}:{v}" for k, v in stats[col].to_dict().items()])
        pdf.multi_cell(0, 10, f"{col}: {values}")
    pdf.ln(3)

    for path, title in chart_paths:
        pdf.add_page()
        pdf.set_font(font_family, "B", 12)
        pdf.set_text_color(*hex_to_rgb(color_hex))
        pdf.cell(200, 10, txt=title, ln=True, align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.image(path, w=180)

    tmp = NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# Anomaly Detection - Basic


def detect_anomalies(df):
    anomalies = pd.DataFrame()
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        flagged = df[(z_scores > 3) | (z_scores < -3)]
        if not flagged.empty:
            flagged["Anomaly Reason"] = f"{col} Z-score outlier"
            anomalies = pd.concat([anomalies, flagged])
    return anomalies


# Anomaly Detection - AI Model (Isolation Forest)


def detect_advanced_anomalies(df):
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) < 1:
        return pd.DataFrame()  # No numeric data to analyze

    data = df[numeric_cols].dropna()

    # ✅ Check if there's any data left after dropna
    if data.empty:
        return pd.DataFrame()

    model = IsolationForest(
        n_estimators=100, contamination=0.05, random_state=42
    )
    model.fit(data)

    predictions = model.predict(data)
    anomaly_indices = data[predictions == -1].index

    anomalies = df.loc[anomaly_indices].copy()
    anomalies["Anomaly Score"] = model.decision_function(
        data.loc[anomaly_indices])
    anomalies["Detected By"] = "Isolation Forest"

    return anomalies.reset_index(drop=True)


if df is not None:
    # 🔍 Anomaly Detection Section
    basic_anomalies = detect_anomalies(df)
    advanced_anomalies = detect_advanced_anomalies(df)

    # Merge and deduplicate
    anomalies = pd.concat(
        [basic_anomalies, advanced_anomalies]).drop_duplicates()

    st.subheader("🚨 Anomaly Detection")
    if not anomalies.empty:
        st.warning(f"{len(anomalies)} anomalies detected using both methods.")
        st.dataframe(anomalies)

        anomaly_excel = convert_df_to_excel(anomalies)
        st.download_button(
            label="⬇️ Download Anomaly Report",
            data=anomaly_excel,
            file_name="anomaly_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.success("✅ No significant anomalies found.")

    # Auto-detect column tags
    column_tags = auto_detect_columns(df)
    st.subheader("🧠 Auto-Detected Column Tags")
    for col, tag in column_tags.items():
        st.markdown(f"**{col}** → {tag}")

    if st.button("Remove Duplicate Rows"):
        df = df.drop_duplicates()
        st.success("✅ Duplicates removed.")

    if st.button("Drop Empty Rows"):
        df = df.dropna(how="all")
        st.success("✅ Empty rows removed.")

    if st.button("Fill Missing Cells with 'N/A'"):
        df = df.fillna("N/A")
        st.success("✅ Missing values filled.")

    st.subheader("🧹 Cleaned Data")
    st.dataframe(df)

    # Auto-detect column tags
    column_tags = auto_detect_columns(df)
    st.subheader("🧠 Auto-Detected Column Tags")
    for col, tag in column_tags.items():
        st.markdown(f"**{col}** → {tag}")

    cleaned_excel = convert_df_to_excel(df)

    st.download_button(
        label="⬇️ Download Cleaned Data as Excel",
        data=cleaned_excel,
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if st.button("📄 Download PDF Summary Report"):
        pdf_path = generate_pdf_summary(
            df,
            custom_title,
            footer_text,
            client_name,
            logo_file,
            selected_font,
            selected_color
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Click to Download PDF Report",
                data=f,
                file_name="cleaned_summary.pdf",
                mime="application/pdf"
            )

    if st.button("🧠 Generate AI Summary"):
        with st.spinner("Thinking..."):
            sample_data = df.head(10).to_csv(index=False)
            prompt = (
                "You are an expert financial analyst.\n\n"
                "Given this data sample in CSV format, provide a clear, concise summary report highlighting:\n"
                "- Number of rows\n"
                "- Key columns\n"
                "- Any patterns, outliers, or interesting stats.\n"
                "- If dates are present, describe date range.\n\n"
                f"CSV Data:\n{sample_data}\n\nSummary:"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You analyze financial datasets and generate professional summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )

            ai_summary = response.choices[0].message.content
            st.subheader("🧠 AI Summary")
            st.write(ai_summary)

    st.subheader("📊 Charts & Visualizations")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    with st.expander("📈 Numeric Column Charts"):
        if len(numeric_cols) >= 1:
            col_to_plot = st.selectbox(
                "Select a numeric column for line chart", numeric_cols)
            st.line_chart(df[col_to_plot])

        if len(numeric_cols) >= 2:
            col1 = st.selectbox("Bar Chart - X-axis", numeric_cols, key="barx")
            col2 = st.selectbox("Bar Chart - Y-axis", numeric_cols, key="bary")
            fig, ax = plt.subplots()
            sns.barplot(x=df[col1], y=df[col2], ax=ax)
            st.pyplot(fig)

    with st.expander("🧮 Categorical Column Charts"):
        if len(cat_cols) >= 1:
            st.subheader("📊 Bar Chart for Categorical Columns")
            bar_col = st.selectbox(
                "Select a column for bar chart", cat_cols, key="catbar")
            bar_data = df[bar_col].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=bar_data.index, y=bar_data.values, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("🥧 Pie Chart")
            pie_col = st.selectbox(
                "Select a column for pie chart", cat_cols, key="piechart")
            pie_data = df[pie_col].value_counts()

            top_n = 6
            if len(pie_data) > top_n:
                top_values = pie_data[:top_n]
                others_sum = pie_data[top_n:].sum()
                pie_data = pd.concat(
                    [top_values, pd.Series({'Others': others_sum})])

            fig1, ax1 = plt.subplots()
            ax1.pie(pie_data, labels=pie_data.index,
                    autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
