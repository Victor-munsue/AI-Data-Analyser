import json
import random
from datetime import datetime, timedelta

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


def generate_otp(email):
    otp = str(random.randint(100000, 999999))
    licenses = load_licenses()
    licenses[otp] = {
        "email": email,
        "expires": (datetime.now() + timedelta(days=30)).isoformat(),
        "activated": False
    }
    save_licenses(licenses)
    return otp


if __name__ == "__main__":
    email = input("Enter customer email: ")
    otp = generate_otp(email)
    print(f"✅ OTP for {email}: {otp} (valid for 30 days, one-time use)")
