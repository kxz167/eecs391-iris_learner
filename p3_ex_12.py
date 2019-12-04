# Plotting
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

color_index = 0
color_choices = ['b', 'r', 'g', 'y']

def plot_flower(input_data, flower_type):

    color_index += 1
    color = color_choices[color_index]
    
    plt.scatter(input_data, width, color=color, label=label)

    
data = genfromtxt('irisdata.csv', delimiter=',')
print(data)