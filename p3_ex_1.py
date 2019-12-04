# Plotting
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import math
from decimal import Decimal

COLORS_INDEX = 0
COLORS_CHOICES = ['b', 'r', 'g', 'y']


def iris_data(file_location):
    target_file = open(file_location, "r")

    petal_raw_data = target_file.readlines()

    petal_data = []

    # Strip newlines
    for i in range(1, len(petal_raw_data)):
        petal_data.append(petal_raw_data[i].strip('\n').split(','))

    return petal_data


def split_classes(petal_data):

    setosa = []
    versicolor = []
    virginica = []

    for point in petal_data:

        flower_type = point[4]

        if flower_type == 'setosa':
            setosa.append(point)

        if flower_type == 'versicolor':
            versicolor.append(point)

        if flower_type == 'virginica':
            virginica.append(point)

    return (setosa, versicolor, virginica)


def numerify(data):
    for flower_set in data:
        for i in range(0, len(flower_set)):
            flower_set[i][0] = float(flower_set[i][0])
            flower_set[i][1] = float(flower_set[i][1])
            flower_set[i][2] = float(flower_set[i][2])
            flower_set[i][3] = float(flower_set[i][3])


def plot_flower(flower_set):

    global COLORS_CHOICES
    global COLORS_INDEX

    color = COLORS_CHOICES[COLORS_INDEX]
    COLORS_INDEX += 1

    label = flower_set[0][4]

    width = []
    length = []

    for point in flower_set:
        length.append(point[2])
        width.append(point[3])

    plt.scatter(length, width, color=color, label=label)
    return (length, width)


def plot_sigmoid():
    x = np.linspace(-5, 5, 100)

    result = 1.0 / (1.0 + np.exp(-x))

    plt.plot(x, result)


def plot_decision_boundary(w_0, w_1, w_2):
    length = np.linspace(0, 10, 100)

    x_1 = length
    width = -1 * (w_0 + x_1 * w_1) / w_2

    plt.plot(length, width)


def plot_surface(w_0, w_1, w_2):

    x = np.outer(np.linspace(0, 8, 100), np.ones(100)).T
    y = np.outer(np.linspace(3, 0, 100), np.ones(100))
    z = 1.0 / (1.0 + np.exp(-(w_0 + w_1 * x + w_2 * y)))

    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_xlim3d(0, 8)
    ax.set_ylim3d(0, 3)
    ax.set_zlim3d(0, 1)

    ax.set_title('Surface Plot for Classification Using Sigmoid Function')
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.set_zlabel('Decision, 0: versicolor, 1:virginica')


def classify(weights,x, y):
    w_0 = weights[0]
    w_1 = weights[1]
    w_2 = weights[2]
    return 1.0 / (1.0 + np.exp(-(w_0 + w_1 * x + w_2 * y)))

def squared_error(weights, data, classes):
    # Get all classifications of the data
    classifications = [[]]* len(data)
    for j in range (0, len(data)):
        for i in range (0, len(data[j][0])):
            classifications[j].append(classify(weights, data[j][0][i], data[j][1][i]))


    # Perform summation
    total = 0
    for i in range(0, len(classifications)):

        # For each classification
        subtotal = 0
        for j in range (0, len(classifications[i])):
            subtotal += Decimal(.5) * Decimal(np.square(classifications[i][j] - classes[i]))
        total += subtotal
        total /= 2
    
    return total

if __name__ == '__main__':

    # Plot Iris Data (1a)
    flower_data = iris_data('irisdata.csv')
    flower_data = split_classes(flower_data)
    numerify(flower_data)

    plt.figure('Versicolor and Virginica')

    versicolor = plot_flower(flower_data[1])
    virginica = plot_flower(flower_data[2])

    plt.title('Iris Data based on length and width')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.xlim(0, 8)
    plt.ylim(0, 3)

    plt.legend()

    # Sigmoid (1b)
    plt.figure('With regression')
    plot_sigmoid()
    plt.title('Standard Sigmoid Function for Soft Decision Boundary')
    plt.ylabel('Decision for classification')

    # Linear Decision Boundary (1c) :
    plt.figure('Versicolor and Virginica')
    plot_decision_boundary(-5.5, .82, 1)

    # Surface Plot (1d):
    plt.figure('Surface Plot')

    plot_surface(-5.5, .82, 1)

    # Test points (1e):
    # Ambiguous
    print('Classifications:')
    print('Versicolor: 5, 1.7: ', classify((-5.5, .82, 1),5,1.7))
    print('Verginica: 5.1, 1.5: ', classify((-5.5, .82, 1),5.1,1.5))

    # Unambiguous
    print('Versicolor: 3.6, 1.3: ', classify((-5.5, .82, 1),3.6,1.3))

    # Mean Squared Error (2a):
    plt.figure('Versicolor and Virginica')
    # Original error (small)
    print('Error for weights (-5.5, .82, 1): ', squared_error((-5.5, .82, 1), (versicolor, virginica), (0, 1)))
    
    # New Error (Big)
    print('Error for weights (-6, .4, 1): ', squared_error((-6, .4, 1), (versicolor, virginica), (0, 1)))
    plot_decision_boundary(-6, .4, 1)

    plt.show()