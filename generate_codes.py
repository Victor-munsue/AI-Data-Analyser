import random
import string
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

# === CONFIGURATION ===
NUM_CODES = 5  # Change this to however many codes you want
DAYS_VALID = 30  # Validity in days

# === GOOGLE SHEETS SETUP ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(
    "my-app-project-465910-e7b33a52d028.json", scopes=SCOPES)  # ✅ This is your actual file
client = gspread.authorize(creds)
sheet = client.open("LicenseKeys").sheet1  # Make sure your sheet exists

# === CODE GENERATION FUNCTIONS ===


def generate_code(length=12):
    return '-'.join(''.join(random.choices(string.ascii_uppercase + string.digits, k=4)) for _ in range(length // 4))


def generate_activation_codes(n=5, days_valid=30):
    codes = []
    for _ in range(n):
        code = generate_code()
        expired_date = (datetime.now() + timedelta(days=days_valid)
                        ).strftime("%Y-%m-%d %H:%M:%S")
        codes.append(["FALSE", code, expired_date, ""])
    return codes


# === GENERATE AND INSERT INTO GOOGLE SHEET ===
new_codes = generate_activation_codes(NUM_CODES, DAYS_VALID)
for row in new_codes:
    sheet.append_row(row)

print(f"✅ {NUM_CODES} new codes added to your LicenseKeys sheet!")
