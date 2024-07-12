import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Function to draw a stylized neutrino wave with higher amplitude on the left
def draw_wave(ax, colour):
    x = np.linspace(0, 11.5, 320)
    y = np.sin(3*x) * 1.9 + 1.9
    y *= (1.0 - x/(2*np.pi))**3
    y += 0.95
    y[100:] = 0.95
    ax.plot(x + 1, y, color=colour, lw=4)

def draw_other_waves(ax, colour_1, colour_2):
    x = np.linspace(0, 3, 100)
    
    y_main = ( np.sin(3*x)) * 1.9 + 1.9
    y_main *= (1.0 - x/(2*np.pi))**3

    # scaling between line1 and line2
    w = (np.sin(2.75*x/3 -1.45) *0.5 +0.5) **1.5

    # more prominent line
    y = (3 - y_main[:100])
    y *= (1 - w)
    y += 0.95
    
    # less prominent
    z = (3 - y_main[:100])
    z *= w
    z += 0.95

    # add some lil wiggles, kinda matter oscillation-y
    h = ( x/3 ) *0.055 * np.sin(50*x)

    ax.plot(x + 1, z*(1-h), color=colour_2, lw=2)
    ax.plot(x + 1, y*(1+h), color=colour_1, lw=3)

# Function to draw a grid, kinda tensor-y
def draw_grid(ax, colour):
    for i in range(1, 5):
        for j in range(1, 5):
            circle = Ellipse((i, j), 0.2, 0.2, color=colour)
            ax.add_patch(circle)

# Define mellow colours
mellow_blue = '#4c96c2'
mellow_green = '#a9db6e'
mellow_orange = '#c28d4c'
mellow_red = '#c2504c'
mellow_purple = '#9983c7' 

# Create a figure
fig, ax = plt.subplots(figsize=(8, 4))


# Draw the adjusted neutrino wave with mellow blue
draw_other_waves(ax, mellow_orange, mellow_red)

# Draw the adjusted neutrino wave with mellow blue
draw_wave(ax, colour=mellow_blue)

# Draw the modern tensor grid with mellow green
draw_grid(ax, colour=mellow_green)

# Add the text with the Greek letter ν in mellow dark blue and larger font size
ax.text(8.45, 2.0, 'νTens', fontsize=92, fontweight='bold', color=mellow_purple, va='center', ha='center', family='sans-serif')

# Set limits and remove axes
ax.set_xlim(0.85, 12.5)      ## ax.set_xlim(0, 9)
ax.set_ylim(0.85, 4.15) ## ax.set_ylim(-0.5, 5.5)
ax.set_aspect(9/9)
ax.axis('off')

# Save the updated logo with the adjustments
plt.savefig('nuTens-logo.png', dpi=300, bbox_inches='tight', transparent=True)

#print("figure size = ", plt.gcf().get_size_inches())

plt.savefig('nuTens-logo-small.png', dpi=50, bbox_inches='tight', transparent=True)
