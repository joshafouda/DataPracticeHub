import matplotlib.pyplot as plt
import numpy as np

# Data for churn visualization
categories = ['Churn', 'No Churn']
values = [30, 70]

# Colors
colors = ['#ff6666', '#66b3ff']

# Explode
explode = (0.1, 0)

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(values, explode=explode, labels=categories, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')  
plt.title('Customer Churn Prediction')

# Save the image
plt.savefig('./imgs/project4/project4.png')

# Show the image
plt.show()

