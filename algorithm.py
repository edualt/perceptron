import random
import numpy as np
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib.pyplot as plt

dataframe = pd.DataFrame()
learning_rate = float()
error_limit = None
weights = []
input_values = [1]
desired_output = []
iteration_count = 0
iteration_numbers = []
weights_history = []
error_values = np.array([])
input_matrix = np.array([])
is_final_iteration = False
error_history = []

def convert_data_to_list(index):
    global input_values, desired_output, input_matrix, is_final_iteration
    data_list = dataframe.values.tolist()

    if input_values:
        input_values.clear()
        input_values.append(1)
    if desired_output:
        desired_output.clear()
    for j in range(len(data_list[index])):
        if j < 3:
            input_values.append(data_list[index][j])
        else:
            desired_output.append(int(data_list[index][j]))
    if is_final_iteration is True:
        input_matrix = np.array([])
    if input_matrix.size == 0:
        input_matrix = np.array([input_values])
    else:
        input_matrix = np.vstack([input_matrix, input_values])

def generate_weights():
    for i in range(len(input_values)):
        # Generate random weights between -2 and 2
        temp_weight = random.uniform(-2, 2)
        weights.append(round(temp_weight, 4))

def calculate_output():
    output = 0
    for i in range(len(input_values)):
        output += input_values[i] * weights[i]
    return output

def apply_activation_function(calculated_output):
    if calculated_output >= 1:
        return 1
    else:
        return 0

def compute_error(calculated_output):
    global error_values, is_final_iteration
    error = desired_output[0] - calculated_output
    if is_final_iteration is True:
        error_values = np.array([])
    if error_values.size == 0:
        error_values = np.array([round(error, 4)])
    else:
        error_values = np.vstack([error_values, round(error, 4)])
    is_final_iteration = False
    return error

def update_weights(input_matrix):
    global weights
    if input_matrix.shape[1] != error_values.shape[0]:
        input_matrix = np.transpose(input_matrix)
    delta_weights = learning_rate * np.dot(input_matrix, error_values)
    temp_weights = np.array([weights])
    temp_weights = temp_weights + np.transpose(delta_weights)
    weights = [round(w, 4) for w in temp_weights[0].tolist()]


def plot_error_graph(iterations, errors):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.plot(iterations, errors, c='red', linewidth=1)
    ax.set(xlabel='Epochs', ylabel='Error',
           title='Error')
    ax.grid()
    ax.set_xlim([0, len(iterations) - 1])
    ax.set_xticks(range(len(iterations)))

def plot_weights(iterations, weights):
    iterationFinal = iterations[-1]
    iteration_copy = iterations.copy()
    iteration_copy.append(iterationFinal + 1)
    markers = ['1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
    colors = ['y', 'c', 'm', 'k','r', 'g', 'b']

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=100)

    for index, values in enumerate(weights):
        for value_index, value in enumerate(values):
            if index < len(weights) - 1:
                next_value = weights[index + 1][value_index]
                ax.plot([iteration_copy[index], iteration_copy[index + 1]], [value, next_value], marker=markers[value_index],
                        color=colors[value_index], label=f"w{value_index}")
            if index == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(0.8, 0.25))

    ax.set(xlabel='Epochs', ylabel='Weights',
           title='Weights')
    ax.set_xlim([0, len(iteration_copy)-1])
    ax.set_xticks(range(len(iteration_copy)))
    ax.grid()

def execute_algorithm():
    global m, is_final_iteration, iteration_count
    m = 0
    while iteration_count < iterations_limit:
        if m == len(dataframe):
            m = 0
            is_final_iteration = True
        while m < len(dataframe):
            convert_data_to_list(m)
            if len(weights) == 0:
                generate_weights()
                weights_history.append(weights)
            calculated_output_temp = round(calculate_output(), 4)
            calculated_output = apply_activation_function(calculated_output_temp)
            compute_error(calculated_output)
            m += 1
        error_norm = np.linalg.norm(error_values)
        error_history.append(error_norm)
        if error_norm >= error_limit:
            update_weights(input_matrix)
        weights_history.append(weights)
        iteration_numbers.append(iteration_count)
        iteration_count += 1
    print("Final weights", weights)

def start_algorithm():
    global dataframe, learning_rate, error_limit, iterations_limit
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename()
    dataframe = pd.read_csv(file_path, delimiter=';', header=None)
    learning_rate = simpledialog.askfloat("Input", "Introduce the learning rate",
                                  parent=root)
    error_limit = simpledialog.askfloat("Input", "Introduce the error threshold",
                                            parent=root)
    iterations_limit = simpledialog.askinteger("Input", "Introduce the number of iterations",
                                            parent=root)
    root.destroy()
    execute_algorithm()
    messagebox.showinfo("Result", "The process has ended")
    plot_error_graph(iteration_numbers, error_history)
    plot_weights(iteration_numbers, weights_history)
    plt.show()
    results_window = tk.Tk()
    initial_weights_label = tk.Label(results_window, text=f"Initial weights: {weights_history[0]}")
    final_weights_label = tk.Label(results_window, text=f"Final weights: {weights_history[-1]}")
    learning_rate_label = tk.Label(results_window, text=f"Learning rate: {learning_rate}")
    error_label = tk.Label(results_window, text=f"Error threshold: {error_limit}")
    epochs_label = tk.Label(results_window, text=f"Epochs: {len(iteration_numbers)}")
    learning_rate_label.pack()
    error_label.pack()
    epochs_label.pack()
    initial_weights_label.pack()
    final_weights_label.pack()
    results_window.mainloop()
root = tk.Tk()
start_button = tk.Button(root, text="Start", command=start_algorithm)
start_button.pack()
root.mainloop()
