"""Greeting Program based on System Timestamp.

Uses the `time` module to extract the current system hour
and prints an appropriate greeting based on standard time intervals.
"""

import time

current_hour = int(time.strftime("%H"))
print(f"Current hour (24-hour format): {current_hour}")

if 5 <= current_hour < 12:
    print("Good morning!")
elif 12 <= current_hour < 17:
    print("Good afternoon!")
elif 17 <= current_hour < 21:
    print("Good evening!")
else:
    print("Good night!")
