import numpy as np
import csv

def read_data_from_csv(filename):
    values = []
    weights = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:
                values.append(int(row[0]))
                weights.append(int(row[1]))
    return values, weights

def knapsack_branch_and_bound(values, weights, max_weight):
    num_projects = len(values)
    best_solution = [0] * num_projects
    best_profit = 0

    def calculate_bound(solution, index, current_weight, current_profit):
        if current_weight > max_weight:
            return -np.inf
        for i in range(index + 1, num_projects):
            if solution[i] == 0 and current_weight + weights[i] > max_weight:
                return current_profit + (values[i] / weights[i]) * (max_weight - current_weight)
        return current_profit

    def branch(solution, index, current_weight, current_profit):
        nonlocal best_solution, best_profit

        if index == num_projects:
            if current_profit > best_profit:
                best_solution = solution.copy()
                best_profit = current_profit
        else:
            solution[index] = 1
            new_weight = current_weight + weights[index]
            new_profit = current_profit + values[index]
            if new_weight <= max_weight and calculate_bound(solution, index, new_weight, new_profit) > best_profit:
                branch(solution, index + 1, new_weight, new_profit)

            solution[index] = 0
            new_weight = current_weight
            new_profit = current_profit
            if calculate_bound(solution, index, new_weight, new_profit) > best_profit:
                branch(solution, index + 1, new_weight, new_profit)

    def recursive_branch(solution, index, current_weight, current_profit):
        if index == num_projects:
            return

        solution[index] = 1
        new_weight = current_weight + weights[index]
        new_profit = current_profit + values[index]
        if new_weight <= max_weight and calculate_bound(solution, index, new_weight, new_profit) > best_profit:
            branch(solution, index + 1, new_weight, new_profit)

        solution[index] = 0
        new_weight = current_weight
        new_profit = current_profit
        if calculate_bound(solution, index, new_weight, new_profit) > best_profit:
            branch(solution, index + 1, new_weight, new_profit)

        recursive_branch(solution, index + 1, current_weight, current_profit)

    solution = [0] * num_projects
    recursive_branch(solution, 0, 0, 0)
    return best_solution, best_profit


max_weight = 200
values, weights = read_data_from_csv("D:/Uvic courses/Optimization/data.csv")


best_solution, best_profit = knapsack_branch_and_bound(values, weights, max_weight)

# results
print("Best solution:", best_solution)
print("Best profit:", best_profit)
