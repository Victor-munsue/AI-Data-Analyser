import streamlit as st

st.set_page_config(page_title="AI Data Analyzer", layout="centered")

# Custom CSS for dark burnt gold theme
st.markdown("""
    <style>
        body {
            background-color: #1a1a1a;
            color: #fff;
        }
        .title-text {
            font-size: 3em;
            color: #daa520;
            text-align: center;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            font-size: 1.4em;
            margin-top: -10px;
            color: #ccc;
        }
        .section-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-top: 30px;
            color: #ffd700;
        }
        .benefits-box {
            background-color: #2e2e2e;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #daa520;
            margin-bottom: 20px;
            color: #f1f1f1;
        }
        .cta-button {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 40px;
            color: #999;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-text">AI Data Analyzer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turn Excel, CSV files into Insights in Seconds</div>',
            unsafe_allow_html=True)

# What it does
st.markdown('<div class="section-title">🔍 What It Does</div>',
            unsafe_allow_html=True)
st.markdown("""
- 📥 Upload Excel or CSV files  
- 🧠 Automatically detects anomalies & suspicious transactions  
- ✨ Cleans, merges, and tags your data  
- 📊 Creates reports with AI-generated summaries  
- 📄 Exports polished PDF and Excel reports  
- 🔐 you don't need Excel or coding skills
""")

# Benefits
st.markdown('<div class="section-title">💡 Who Is It For?</div>',
            unsafe_allow_html=True)
st.markdown("""
<div class="benefits-box">
    <strong>✅ Accountants & Financial Analysts:</strong><br>
    Save hours of manual work. Deliver AI-powered reports instantly.
</div>
<div class="benefits-box">
    <strong>✅ Business Owners:</strong><br>
    Track your money. Audit and report your business finances with ease.
</div>
<div class="benefits-box">
    <strong>✅ Freelancers & Agencies:</strong><br>
    Offer financial analysis as a service with no coding needed.
</div>
<div class="benefits-box">
    <strong>✅ Students & Beginners:</strong><br>
    Generate insights, automate reports, and grow your data skills.
</div>
""", unsafe_allow_html=True)

# How to Get Access
st.markdown('<div class="section-title">💰 How to Get Access</div>',
            unsafe_allow_html=True)
st.markdown("""
Access is just **$10/month** (₦15,000) – includes AI tools, PDF export & unlimited data cleaning.

💳 **Pay once, get a unique password instantly.**

You can pay through **OPay (international-friendly)**:

**Bank Name:** OPay  
**Account Number:** 9161398285  
**Account Name:** VICTOR CHUKWUNOMUNSUE CHUKWUJINDU  

After payment, click below to send your proof on WhatsApp and receive your OTP key. 
You can also use that same button for inquiries""")

# WhatsApp CTA
whatsapp_number = "+2349161398285"
message = "Hello, I just paid for AI Financial Analyzer. Here's my payment proof:"
link = f"https://wa.me/{whatsapp_number}?text={message.replace(' ', '%20')}"

st.markdown(f"""
<div class="cta-button">
    <a href="{link}" target="_blank">
        <button style="padding: 15px 40px; font-size: 18px; background-color: #daa520; color: black; border: none; border-radius: 8px; cursor: pointer;">
            🔓 Send Payment Proof & Get OTP
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">🔐 Built with 💡 by Yahnova Automations • All rights reserved</div>',
            unsafe_allow_html=True)
