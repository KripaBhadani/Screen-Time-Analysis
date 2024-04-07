import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Screentime - App Details.csv")

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Print basic information about the data
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.columns)
# Find the app with the most unlocks and related notifications
most_unlocks = df['TimesOpened'].max()
corelated_notifications = df.loc[df['TimesOpened'] == most_unlocks, 'Notifications'].iloc[0]
print("No. of times user got notifications = ", corelated_notifications)
print("Max number of unlocks to their phone =", most_unlocks)

# Find the app with the most notifications and related unlocks
most_notifications = df['Notifications'].max()
corelated_unlocks = df.loc[df['Notifications'] == most_notifications, 'TimesOpened'].iloc[0]
print("No. of times user unlocked phone = ", corelated_unlocks)
print("No. of max notifications =", most_notifications)

# Visualize data using line plots
plt.figure(figsize=(12, 15))
df.plot(subplots=True, ax=plt.gca())
plt.xlabel('Values')
plt.show()

# Sort data by 'Usage' and plot top 20
sorted_df = df.sort_values(by=['Usage'], ascending=False)
plt.figure(figsize=(15, 8))
sns.barplot(x='Date', y='Usage', data=sorted_df.head(20), hue='Date', legend=False)
plt.xticks(rotation=60)
plt.ylabel('Usage')
plt.xlabel('Date')
plt.title('Total Usage per Day')
plt.show()

# Find app opened max times
max_times = df['TimesOpened'].max()
used_app = df.loc[df['TimesOpened'] == max_times, 'App'].iloc[0]
print(used_app, 'was opened max times ~', max_times, 'times')

# Find app with maximum usage
max_usage = df['Usage'].max()
_app = df.loc[df['Usage'] == max_usage, 'App'].iloc[0]
print(_app, 'had maximum usage ~', max_usage)

# Find app with max notifications and least notifications
max_notification = df['Notifications'].max()
u_app = df.loc[df['Notifications'] == max_notification, 'App'].iloc[0]
print(u_app, 'had max notifications ~', max_notification)

min_notification = df['Notifications'].min()
us_app = df.loc[df['Notifications'] == min_notification, 'App'].iloc[0]
print(us_app, 'had least notifications ~', min_notification)

# Plot total times apps were opened per day
plt.figure(figsize=(15, 8))
sns.barplot(x='Date', y='TimesOpened', data=df.head(20), hue='Date', legend=False)
plt.xticks(rotation=60)
plt.ylabel('TimesOpened')
plt.xlabel('Date')
plt.title('Total times apps were opened per Day')
plt.show()

# Count occurrences of Instagram and WhatsApp and check if they are the same
instagram_count = (df['App'] == 'Instagram').sum()
whatsapp_count = (df['App'] == 'WhatsApp').sum()

print("Number of occurrences for Instagram:", instagram_count)
print("Number of occurrences for WhatsApp:", whatsapp_count)

if instagram_count == whatsapp_count:
    print("The number of occurrences for Instagram and WhatsApp is the same.")
else:
    print("The number of occurrences for Instagram and WhatsApp is not the same.")

# Feature Engineering
# Create 'Weekday' and 'Weekend' features
df['Weekday'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
df['Weekend'] = df['Weekday'].isin([5, 6]).astype(int)  # 1 if weekend, 0 if weekday

# Calculate average usage per day for each app
avg_usage_per_day = df.groupby(['App', 'Date']).agg({'Usage': 'sum'}).reset_index().groupby('App')['Usage'].mean()
df['AvgUsagePerDay'] = df['App'].map(avg_usage_per_day)

# Calculate average usage per session for each app
df['AvgUsagePerSession'] = df['Usage'] / df['TimesOpened']

# Session Length
df['SessionLength'] = df['Usage'] / 60  # Convert seconds to minutes

# Interactions
df['Interactions'] = df['TimesOpened'] + df['Notifications']

# Daily Trends
df['DailyUsageTrend'] = df.groupby('Date')['Usage'].transform('mean')

# Weekday vs. Weekend Usage
df['Weekday'] = df['Date'].dt.dayofweek < 5  # Weekday: True, Weekend: False

# User Engagement Score
df['EngagementScore'] = (df['Usage'] + df['Interactions'] + df['SessionLength']) / 3

# Visualization of Feature Engineering

# Distribution of Weekday vs. Weekend Usage
plt.figure(figsize=(10, 6))
sns.barplot(x='App', y='Usage', hue='Weekend', data=df, palette={0: 'lightblue', 1: 'lightgreen'})
plt.title('Distribution of Weekday vs. Weekend Usage')
plt.xlabel('App')
plt.ylabel('Usage')
plt.xticks(rotation=45)
plt.legend(title='Weekend', loc='upper right')
plt.show()

# Average Usage per Day
plt.figure(figsize=(10, 6))
sns.barplot(x='App', y='AvgUsagePerDay', data=df, palette='viridis')
plt.title('Average Usage per Day for Each App')
plt.xlabel('App')
plt.ylabel('Average Usage per Day')
plt.xticks(rotation=45)
plt.show()

# Average Usage per Session
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='AvgUsagePerSession', hue='App', bins=20, kde=True, palette={'Instagram': 'pink', 'WhatsApp': 'green', 'Snapchat': 'yellow', 'LinkedIn': 'blue'})
plt.title('Distribution of Average Usage per Session for Each App')
plt.xlabel('Average Usage per Session')
plt.ylabel('Frequency')
plt.legend(title='App')
plt.show()

# Session Length Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='SessionLength', bins=20, kde=True)
plt.title('Distribution of Session Length')
plt.xlabel('Session Length (minutes)')
plt.ylabel('Frequency')
plt.show()


# Daily Trends
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='DailyUsageTrend', data=df, estimator='mean')
plt.title('Daily Usage Trends')
plt.xlabel('Date')
plt.ylabel('Average Usage')
plt.xticks(rotation=45)
plt.show()

# Weekday vs. Weekend Usage
plt.figure(figsize=(10, 6))
sns.barplot(x='App', y='Usage', hue='Weekday', data=df, palette={True: 'lightblue', False: 'lightgreen'})
plt.title('Weekday vs. Weekend Usage')
plt.xlabel('App')
plt.ylabel('Usage')
plt.xticks(rotation=45)
plt.legend(title='Weekday')
plt.show()

# User Engagement Score
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='EngagementScore', bins=20, kde=True)
plt.title('Distribution of User Engagement Score')
plt.xlabel('Engagement Score')
plt.ylabel('Frequency')
plt.show()

# Plot relationship between Notifications and Usage
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Notifications", y="Usage", hue="App", palette={'Instagram': 'pink', 'WhatsApp': 'green', 'Snapchat': 'yellow', 'LinkedIn': 'blue'})
plt.title("Relationship between number of Notifications and Usage")
plt.show()
