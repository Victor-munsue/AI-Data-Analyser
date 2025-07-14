import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
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

# Hybrid Payment Form Section (to insert into your app before activation)
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import tempfile
import os

# --- CONFIG ---
DRIVE_FOLDER_ID = "1Zr4sRRInBdEWhtTgnwN_f0gZCoHkD6p7"
WHATSAPP_NUMBER = "+2349161398285"

# --- Setup Google API ---
creds = Credentials.from_service_account_info(st.secrets["gspread"])
sheet_client = gspread.authorize(creds)
drive_service = build("drive", "v3", credentials=creds)

# --- Create Payments Sheet if Missing ---


def get_or_create_payment_sheet():
    sheet = sheet_client.open("LicenseKeys")
    try:
        payments = sheet.worksheet("Payments")
    except:
        payments = sheet.add_worksheet("Payments", rows=1000, cols=5)
        payments.append_row(["Timestamp", "Name", "Email", "Screenshot Link"])
    return payments

# --- Upload to Drive Folder ---


def upload_to_drive(uploaded_file, name):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_file.read())
    tmp.flush()

    media = MediaFileUpload(tmp.name, resumable=True)
    file_metadata = {
        "name": f"{name}-{datetime.now().strftime('%Y%m%d%H%M%S')}.png",
        "parents": [DRIVE_FOLDER_ID]
    }
    file = drive_service.files().create(
        body=file_metadata, media_body=media, fields="id").execute()
    file_id = file.get("id")
    os.unlink(tmp.name)
    return f"https://drive.google.com/file/d/{file_id}/view"


# --- Show Payment Form ---
st.markdown("### 💳 Submit Payment Proof")
with st.form("payment_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    screenshot = st.file_uploader(
        "Upload Payment Screenshot", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Send & Notify on WhatsApp")

if submit:
    if not name or not email or not screenshot:
        st.warning("Please complete all fields before submitting.")
    else:
        # Upload file
        file_url = upload_to_drive(screenshot, name)
        # Save to sheet
        sheet = get_or_create_payment_sheet()
        sheet.append_row([str(datetime.now()), name, email, file_url])

        # Redirect to WhatsApp
        message = f"Hello, I just paid for AI Analyzer. Name: {name}, Email: {email}. Kindly confirm."
        encoded_msg = message.replace(" ", "%20")
        whatsapp_link = f"https://wa.me/{WHATSAPP_NUMBER}?text={encoded_msg}"
        st.success("✅ Submitted! Click below to notify us on WhatsApp.")
        st.markdown(
            f"[🔔 Notify on WhatsApp]({whatsapp_link})", unsafe_allow_html=True)
        st.stop()


def get_worksheet():
    creds = Credentials.from_service_account_info(st.secrets["gspread"])
    client = gspread.authorize(creds)
    # Must match your Google Sheet name exactly
    sheet = client.open("LicenseKeys")
    return sheet.sheet1


def validate_code(code_input):
    sheet = get_worksheet()
    records = sheet.get_all_records()

    for i, row in enumerate(records):
        if str(row["Code"]).strip() == code_input.strip():
            if str(row["Activated"]).upper() == "TRUE":
                return None, "🔒 Code already used"
            if datetime.now() > datetime.strptime(row["Expired"], "%Y-%m-%d %H:%M:%S"):
                return None, "⌛ Code expired"

            # ✅ Mark as activated in Google Sheet
            sheet.update_cell(i + 2, 1, "TRUE")  # 'Activated' column = TRUE
            return True, "✅ Code activated for 30 days"

    return None, "❌ Invalid code"


SESSION_KEY = "access_until"


def is_session_active():
    if SESSION_KEY in st.session_state:
        return datetime.now() < st.session_state[SESSION_KEY]
    return False


def set_session_30_days():
    st.session_state[SESSION_KEY] = datetime.now() + timedelta(days=30)


if not is_session_active():
    st.sidebar.header("🔐 Secure Access")
    code_input = st.sidebar.text_input(
        "Enter Activation Code", type="password")

    if code_input:
        status, msg = validate_code(code_input)
        st.sidebar.info(msg)

        if status:
            set_session_30_days()
            st.experimental_rerun()
        else:
            st.stop()
    else:
        st.stop()


# Initialize OpenAI client

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


st.set_page_config(page_title="Data Analyzer", layout="wide")
st.title("📊 Data Analyzer - AI Enhanced")

# 🔐 Secure Access
st.sidebar.header("🔐 Secure Access")


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
