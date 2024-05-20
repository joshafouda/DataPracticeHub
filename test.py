import matplotlib.pyplot as plt

# Create a figure
fig, ax = plt.subplots(figsize=(6, 1.5))

# Set the background color
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Hide axes
ax.axis('off')

# Add the text
ax.text(0.5, 0.5, "DataPracticeHub\nMade by Josué AFOUDA", 
        verticalalignment='center', horizontalalignment='center',
        color='white', fontsize=20, weight='bold', family='monospace')

# Save the image
plt.savefig('imgs/DataPracticeHub_logo.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# Show the image
plt.show()
