import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Load required environment variables
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO", EMAIL_USER)  # Default to sending to self if not provided

def send_email():
    # Read pattern file
    pattern_file = Path("logs/current_detected_patterns.txt")
    if not pattern_file.exists():
        print("No pattern file found to send.")
        return

    with open(pattern_file, "r") as f:
        pattern_content = f.read()

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO
    msg["Subject"] = "📈 SPX Pattern Alert (Auto Generated)"

    msg.attach(MIMEText(pattern_content, "plain"))

    # Send the email using Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

if __name__ == "__main__":
    send_email()
