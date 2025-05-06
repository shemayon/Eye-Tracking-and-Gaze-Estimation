import os
import subprocess
from datetime import datetime, timedelta
import random

# Repository path
repo_path = r"C:\Users\shema\OneDrive\Desktop\EG tracker\gt_git\Eye-Tracking-and-Gaze-Estimation"

# Start and end dates
start_date = datetime(2024, 10, 10)
end_date = datetime(2025, 5, 7)

# Month name mapping
month_names = {
    1: "january", 2: "february", 3: "march", 4: "april", 5: "may",
    6: "june", 7: "july", 8: "august", 9: "september",
    10: "october", 11: "november", 12: "december"
}

# Build dates by month name
dates_by_month_name = {}

current_date = start_date
while current_date <= end_date:
    month_key = month_names[current_date.month]
    if month_key not in dates_by_month_name:
        dates_by_month_name[month_key] = []
    if current_date.day not in dates_by_month_name[month_key]:
        dates_by_month_name[month_key].append(current_date.day)
    current_date += timedelta(days=2)  # alternate days

# Go to repo
os.chdir(repo_path)

# Create commits
for month_key, days in dates_by_month_name.items():
    month_num = [k for k, v in month_names.items() if v == month_key][0]
    for day in days:
        try:
            date = datetime(2024 if month_num >= 10 else 2025, month_num, day)
        except ValueError:
            continue  # skip invalid days like Feb 30

        num_commits = random.randint(5, 7)

        for _ in range(num_commits):
            filename = f"dummy_{date.strftime('%Y%m%d')}_{random.randint(1, 100)}.txt"
            with open(filename, "w") as f:
                f.write(f"Commit on {date.strftime('%Y-%m-%d %H:%M:%S')}")

            subprocess.run(["git", "add", filename])
            commit_message = f"Commit on {date.strftime('%Y-%m-%d %H:%M:%S')}"
            env = os.environ.copy()
            env["GIT_COMMITTER_DATE"] = date.strftime("%Y-%m-%d %H:%M:%S")
            env["GIT_AUTHOR_DATE"] = date.strftime("%Y-%m-%d %H:%M:%S")
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                env=env,
            )

            os.remove(filename)
