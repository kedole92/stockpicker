
# is_trading_day.py
import sys, os, datetime, pytz

IST = pytz.timezone("Asia/Kolkata")
today_ist = datetime.datetime.now(IST).date()

# weekend?
if today_ist.weekday() >= 5:  # 5=Sat, 6=Sun
    print("NOT_TRADING_DAY")
    sys.exit(0)

# custom holidays
holidays_path = "market_holidays.txt"
if os.path.exists(holidays_path):
    with open(holidays_path, "r") as f:
        hols = {ln.strip() for ln in f if ln.strip()}
    if today_ist.isoformat() in hols:
        print("NOT_TRADING_DAY")
        sys.exit(0)

print("TRADING_DAY")
