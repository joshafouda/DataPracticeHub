import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Ensure the imgs directory exists
os.makedirs('imgs', exist_ok=True)

# Generate random samples for the scatter plot
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 3 * x + np.random.randn(100) * 5

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust height to prevent the animation from being too tall
line, = ax.plot([], [], 'r-', lw=2)
text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Function to update the scatter plot and regression line
def update(frame, x, y, line, text):
    plt.cla()  # Clear the current plot
    
    # Scatter plot
    plt.scatter(x[:frame], y[:frame], color='blue', s=10)
    
    if frame > 1:
        # Fit a linear regression model to the current data
        b, m = np.polyfit(x[:frame], y[:frame], 1)
        
        # Plot regression line
        plt.plot(x[:frame], b + m * x[:frame], 'r-')
        
        # Update the equation text
        equation = f'y = {m:.2f}x + {b:.2f}'
        text.set_text(equation)
        
    plt.xlim(0, 10)
    plt.ylim(min(y) - 5, max(y) + 5)
    #plt.title("DataPracticeHub\nMade by Josué AFOUDA")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x), fargs=(x, y, line, text), interval=50, blit=False)

# Save the animation as a GIF using PillowWriter
animation_path = os.path.join('imgs', 'logo_animation.gif')
ani.save(animation_path, writer=PillowWriter(fps=20))

plt.close(fig)  # Close the figure to prevent it from displaying in the notebook

print(f"Animation saved to {animation_path}")
