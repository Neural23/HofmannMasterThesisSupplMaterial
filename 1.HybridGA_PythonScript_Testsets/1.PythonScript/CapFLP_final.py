

"""


Python Script for Solving the Capacitated Facility Location Problem (CFLP) with a Hybride Framework that
utilized a Parallelized Genetic Algorithm (NSGA-II) with Linear Programming integrated in its Fitness Function

The Hybrid Framework implemented in this Python Script utilizes a Genetic Algorithm (GA) to optimize the
facility configuration. Due to the fact that the minimization of total costs when using a single-objective 
algorithm would be accompanied by a minimization of the number of facilities and thus a minimization of 
demand coverage, the NSGA-II is used to maximize total costs while concurrently maximizing demand coverage. 
To determine the minimum total variable costs within the Fitness Function of the GA, Linear Programming is 
used to solve the classical transportation problem (also known as Hitchcock-Koopmans transportation model). 
If you posses a valid software license for CPLEX, please follow the steps provided in the corresponding GitHub 
repository to use the CPLEX Solver within the Fitness Function. Otherwise, the CBC Solver from COIN-OR will 
be used. To accelerate the algorithm and reduce CPU Time, the number of available processes (corresponding 
with available CPU cores on your computer) is automatically detected to execute the time consuming function
in parallel. As benchmark instances, the test sets from the OR Library, introduced by J.E. Beasley in 1988 
are used. For further and detailed informations, please visit the GitHub Respository at the URL provided below.


Usage:
To execute the script, run it from the command line or an integrated development environment (IDE).
The user will be prompted to select a dataset and specify fixed costs, or choose an automated execution mode.

Execute the following command in terminal to install required libraries before running the code for the first time
pip install -r requirements.txt


The following code is implemented with the aid of the DEAP Framework (Fortin et al., 2012)
Documentation on DEAP is provided on the official DEAP website: https://deap.readthedocs.io/en/master/


Author: Martin Hofmann
Contribution: Master's Thesis at TH Aschaffenburg
Date: July 2024

GitHub Repository: https://github.com/Neural23/HofmannMasterThesisSupplementaryMaterial
Website: https://deepcodelines.com


"""




# Import required libraries
import numpy as np
import pulp
import random
import sys
import os
import time
import multiprocessing
import pandas as pd
from deap import base, creator, tools





def user_query():
    
    """
    Prompts the user to select a dataset and its corresponding fixed costs.
    
    Returns:
    tuple
        dataset : str
            The selected dataset identifier.
        fixcosts_value : str or None
            The selected fixed costs per facility or None for specific datasets.
    """


    dataset_number_mapping = {
        'IV': '4', 'V': '5', 'VI': '6', 'VII': '7',
        'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11',
        'XII': '12', 'XIII': '13'
    }
    
    valid_fixcosts_options = ['7500', '12500', '17500', '25000']

    
    # Ensure that a valid dataset is selected with case-insensitive input.
    while True:
        dataset = input("Options of test sets: IV, V, VI, VII, VIII, IX, X, XI, XII, XIII, a, b, c\nPlease select the desired set from the selection shown above: ").upper()
        if dataset in dataset_number_mapping or dataset in ['A', 'B', 'C']:
            break
        else:
            print("Invalid dataset selected. Please try again.")
            
            
    # Skip fixed costs input for datasets 'A', 'B', 'C'
    if dataset in ['A', 'B', 'C']:
        return dataset, None       
            
    
    
    if dataset == 'V':
        # Special logic for dataset 'V' to ensure '17500' is the only accepted input for fixed costs.
        while True:
            fixcosts_value = input("\nOptions of Fixcosts for dataset 'V': 17500\nPlease select the fixed costs per facility: ")
            if fixcosts_value == '17500':
                break
            else:
                print("Invalid fixed costs selected for dataset 'V'. '17500' is the only valid option. Please try again.")

    else:
        # Ensure that a valid fixed costs value is selected with case-insensitive input for other datasets.
        while True:
            fixcosts_value = input("\nOptions of Fixcosts: 7500, 12500, 17500, 25000\nPlease select the fixed costs per facility: ").upper()
            if fixcosts_value in valid_fixcosts_options:
                break
            else:
                print("Invalid fixed costs selected. Please try again.")
    
    return dataset, fixcosts_value





def select_dataset_path(dataset, fixcosts_value):

    """
    Determines the file path based on the selected dataset and fixed costs.
    
    Parameters:
    dataset : str
        The selected dataset identifier.
    fixcosts_value : str or None
        The selected fixed costs per facility or None for specific datasets.
    
    Returns:
    filepath: str
        The constructed file path based on the dataset and fixed costs.
    
    Raises:
    ValueError
        If the fixed costs value is not provided for datasets IV to XIII.
        If an invalid dataset is selected.
    """


    dataset_number_mapping = {
        'IV': '4', 'V': '5', 'VI': '6', 'VII': '7',
        'VIII': '8', 'IX': '9', 'X': '10', 'XI': '11',
        'XII': '12', 'XIII': '13'
    }

    path_prefix_mapping = {
        'IV': 1, 'V': 2, 'VI': 3, 'VII': 4,
        'VIII': 5, 'IX': 6, 'X': 7, 'XI': 8,
        'XII': 9, 'XIII': 10
    }

    # Automatically determine the base path from the current notebook directory
    base_path = os.path.join(os.getcwd(), 'data')
    fixcosts_to_number = {'7500': '1', '12500': '2', '17500': '3', '25000': '4'}

    # Special logic for dataset 'V'
    if dataset == 'V' and fixcosts_value == '17500':
        fixcosts_number = '1'  # Specific mapping for '17500' in the context of dataset 'V'
    else:
        fixcosts_number = fixcosts_to_number.get(fixcosts_value, '')

    if dataset in dataset_number_mapping:
        if fixcosts_value is None:
            raise ValueError("Fixed costs value must be provided for datasets IV to XIII.")
        
        dataset_number = dataset_number_mapping[dataset]
        path_prefix = path_prefix_mapping[dataset]
        file_path = os.path.join(base_path, f"{path_prefix}.{dataset}", f"cap{dataset_number}{fixcosts_number}_{dataset}_{fixcosts_value}.txt")
    elif dataset in ['A', 'B', 'C', 'a', 'b', 'c']:
        dataset = dataset.lower()
        file_path = os.path.join(base_path, "11.ABC", f"cap{dataset}.txt")
    else:
        raise ValueError("Invalid dataset selected")

    print("\nfile path: ", file_path, "\n")
    return file_path





def execution_mode():

    """
    Prompts the user to choose between running the Genetic Algorithm (GA) once for a 
    specific dataset or automatically for all datasets IV to XIII or automatically for
    all large datasets 'A','B' and 'C'.
    
    Returns:
    meta_choice: str
        The user's choice for execution mode ('single', 'auto', or 'autolarge').
    """
        

    # Valid options for the execution mode
    valid_choices = ['single', 'auto', 'autolarge']
    
    # Continuously prompt the user for their choice until a valid input is provided. 
    while True:
        meta_choice = input("Enter 'single' to run the algorithm on a specific dataset or enter 'auto' to run the\n" + 
                            "algorithm automatically for all datasets from 'IV-7500' to 'XIII-25000' in sequence.\n" +
                            "Enter 'autolarge' to run the algorithm automatically for a specific set of capacities for datasets a, b, c.\n").lower()
        
        if meta_choice in valid_choices:
            break
        else:
            print("Invalid selection. Please enter 'single', 'auto', or 'autolarge'.")
    
    # Return the user's choice for further processing.
    return meta_choice





def read_data_create_matrix(dataset, fixcosts_value, capacity=None):

    """
    Reads data from a file (test instances from the OR Library, presented by J.E. Beasley in 1988) 
    and creates the necessary data structures (costmatrix and so on) for the algorithm.
    
    Parameters:
    dataset : str
        The selected dataset identifier.
    fixcosts_value : str or None
        The selected fixed costs per facility or None for specific datasets.
    capacity : int, optional
        The selected capacity for datasets A, B, or C, default is None.
    
    Returns:
    tuple that contains
        warehouses : int
            Number of warehouses.
        customers : int
            Number of customers.
        list_supply : list of float
            List of capacities for each warehouse.
        list_fixcosts : list of float
            List of fixed costs for each warehouse.
        cost_matrix : dict of numpy arrays
            Cost matrix where each element corresponds to the shipping costs per unit (float)
            from warehouse i (warehouse_index:row index) to customer j (customer_index: column index)
        list_demand : list of float
            List of demands for each customer.
        meta_penalty : float
            Penalty value to penalize and therefore effectively exclude individuals with insufficient demand coverage
            from the optimization process.
    """


    file_path = select_dataset_path(dataset, fixcosts_value)

    # In case that data set A, B or C was selected
    if fixcosts_value is None: 
        available_capacities = None

        # Determine the dataset and set available capacities based on the file name
        if 'capa.txt' in file_path:
            available_capacities = [8000, 10000, 12000, 14000]
        elif 'capb.txt' in file_path:
            available_capacities = [5000, 6000, 7000, 8000]
        elif 'capc.txt' in file_path:
            available_capacities = [5000, 5750, 6500, 7250]

        # If capacity is provided and available, select it, otherwise prompt the user
        if capacity and capacity in available_capacities:
            selected_capacity = capacity
        else:
            if available_capacities:
                print("Available capacities: " + ", ".join(map(str, available_capacities)))
                while True:
                    try:
                        selected_capacity = int(input("Please select a capacity: "))
                        if selected_capacity in available_capacities:
                            break
                        else:
                            print("Invalid capacity selected. Please try again.")
                    except ValueError:
                        print("Please enter a numerical value.")
            else:
                selected_capacity = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    warehouses, customers = map(int, lines[0].split())

    warehouse_data = []

    for i in range(1, warehouses + 1):
        line_data = lines[i].split()

        # Check if the capacity is specified as "capacity" and replace it with the selected capacity
        if line_data[0] == "capacity":
            if selected_capacity is not None:
                capacity = selected_capacity
            else:
                print("Error: Capacity placeholder found but no capacity selected.\n")
                print("Terminate program.")
                return
        else:
            capacity = float(line_data[0])
        fixed_cost = float(line_data[1])
        warehouse_data.append((capacity, fixed_cost))

    remaining_lines = lines[warehouses + 1:]
    all_numbers = [float(number) for line in remaining_lines for number in line.split()]

    cost_matrix = {}

    group_size = warehouses + 1
    list_demand = []

    for i in range(0, len(all_numbers), group_size):
        customer_index = i // group_size
        customer_demand = all_numbers[i]
        list_demand.append(customer_demand)

        for j in range(1, group_size):
            warehouse_index = j - 1
            variable_cost = all_numbers[i + j]
            cost_per_unit = variable_cost / customer_demand if customer_demand != 0 else 0

            if warehouse_index not in cost_matrix:
                cost_matrix[warehouse_index] = np.zeros(customers)
            cost_matrix[warehouse_index][customer_index] = cost_per_unit

    # Initialize the lists for supply (capacities) and fixed costs
    list_supply = []
    list_fixcosts = []

    # Separate supplies (capacities) and fixed costs into separate lists
    for capacity, fixcost in warehouse_data:
        list_supply.append(capacity)
        list_fixcosts.append(fixcost)

    if isinstance(cost_matrix, dict):
        cost_matrix1 = np.array([cost_matrix[key] for key in sorted(cost_matrix.keys())])

    # variable for penalizing individuals at genetic algorithm level
    meta_penalty = (sum(list_fixcosts) + np.max(cost_matrix1) * sum(list_demand)) * 2
    # print(list_fixcosts)
    # print(list_supply)


    return warehouses, customers, list_supply, list_fixcosts, cost_matrix, list_demand, meta_penalty





def customize_params():

    """
    Prompts the user to specify custom parameters for the algorithm or use default values.
    
    The function interacts with the user through the console to decide whether to use default 
    parameters or custom ones for the genetic algorithm.
    
    Returns:
    custom_paramslist:
        List [max_gen (int), pop_size (int), cx_pb (float), mut_pb(float), cut_off (float)]
        of custom parameters or an empty list if default values are chosen.
    """
    

    # Ensure that a valid decision is made with case-insensitive input.
    while True:
        decide_customize = input(
            "Would you like to specify the parameters 'maximum number of generations' (max_gen), 'population size' (pop_size), 'crossover probability' (cx_pb), 'mutation probability' (mut_pb), and 'cut-off value (cut_off)' yourself, or should the algorithm execute with the default values of pop_size = 100, max_gen = 100, cx_pb = 0.9, mut_pb = 0.03?\n\n"
            "You have the following options:\n\n"
            "Enter '0' to run the algorithm with the default values.\n"
            "Enter '1' to set the parameters yourself.\n"
        )
        if decide_customize in ['0', '1']:
            break
        else:
            print("Invalid input. Please enter '0' or '1'.")
    
    # Initialize the list for custom parameters
    custom_paramslist = []
    
    if decide_customize == '0':
        return custom_paramslist
    
    else:
        # Prompt the user to input custom values for each parameter
        while True:
            try:
                max_gen = int(input("Enter the maximum number of generations (max_gen): "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer value.")
        
        while True:
            try:
                pop_size = int(input("Enter the population size (pop_size): "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer value.")
        
        while True:
            try:
                cx_pb = float(input("Enter the crossover probability (cx_pb), a float value between 0 and 1: "))
                if 0 <= cx_pb <= 1:
                    break
                else:
                    print("Invalid input. Please enter a value between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a float value.")
        
        while True:
            try:
                mut_pb = float(input("Enter the mutation probability (mut_pb), a float value between 0 and 1: "))
                if 0 <= mut_pb <= 1:
                    break
                else:
                    print("Invalid input. Please enter a value between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a float value.")
        
        while True:
            try:
                cut_off = float(input("Enter the cut-off value (cut_off), a float value between 0 and 100: "))
                if 0 <= cut_off <= 100:
                    break
                else:
                    print("Invalid input. Please enter a value between 0 and 100.")
            except ValueError:
                print("Invalid input. Please enter a float value.")
        
        # Append the custom values to the list
        custom_paramslist = [max_gen, pop_size, cx_pb, mut_pb, cut_off]
    
    return custom_paramslist





def get_balanced(supply, demand, costs, penalty_costs=10e10):
    
    """
    Balances the total supply and total demand in the event that the total supply is not equal to the total demand
    by adding an artificial supply node or an artificial demand node whose demand or supply quantity equals the 
    difference between total demand and total supply. In this event, the cost matrix is also adjusted accordingly 
    by adding penalty costs to the artificial supply node or artificial demand node.
    
    Parameters:
    supply : list of float
        List of supply quantities for each supply node.
    demand : list of float
        List of demand quantities for each demand node.
    costs : 2D numpy array of floats
        Cost matrix where each element corresponds to the shipping costs per unit (float)
        from warehouse i (row index) to customer j (column index)
    penalty_costs : float, optional
        Penalty cost for the artificial supply or demand node, default is 10e10.
    
    Returns:
    tuple
        supply : list of int
            List of supply quantities for each supply node.
        demand : list of int
            List of demand quantities for each demand node.
        costs : list of lists of floats
            Cost matrix where each element corresponds to the shipping costs per unit (float)
            from warehouse i (row index) to customer j (column index)
        artificial_costs : float
            Total artificial costs incurred due to the added artificial supply or demand node
        new_supply : list of int
            Updated list of supply quantities including the supply quantity of the added artificial supply node.
        new_demand : list of int
            Updated list of demand quantities including the demand quantity of the added artificial demand node.
        new_costs : list of lists of int
            Updated cost matrix where each element corresponds to the shipping costs per unit (float)
            from warehouse i (row index) to customer j (column index), including the artificial penalty 
            costs per unit between the added artificial supply node and each demand node or between the 
            added artificial demand node and each supply node, respectively.
    
    Raises:
    Exception
        If the total supply is less than total demand and penalty_costs is not provided.
    """


    # Calculate total quantites of supply and demand
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    # Ensure costs is always a list
    if not isinstance(costs, list):
        costs = costs.tolist()
    
    artificial_costs = 0
        
    # Add artificial supplier if demand exceeds supply
    if total_supply < total_demand:
        if penalty_costs is None:
            raise Exception('Supply less than demand, penalty_costs required')
        extra_supply = total_demand - total_supply
        new_supply = supply + [extra_supply]
        new_costs = costs + [[penalty_costs for _ in range(len(demand))]]
        artificial_costs = extra_supply * penalty_costs
        
        return new_supply, demand, new_costs, artificial_costs
    
    # Add artificial demander if supply exceeds demand
    if total_supply > total_demand:
        extra_demand = total_supply - total_demand
        new_demand = demand + [extra_demand]
        new_costs = [row + [0] for row in costs]

        return supply, new_demand, new_costs, artificial_costs
     
    return supply, demand, costs, artificial_costs





def transportation_method(supply, demand, costs, solver_type):
    
    """
    Solves the transportation problem using linear programming and returns the total 
    cost of transportation (Total Variable Costs)
    
    Parameters:
    supply : list of int
        List of supply quantities for each supply node.
    demand : list of int
        List of demand quantities for each demand node.
    costs : 2D numpy array of floats
        Cost matrix where each element corresponds to the shipping costs per unit (float)
        from warehouse i (row index) to customer j (column index)
    solver_type: str
        The type of solver to use, "CBC" or "CPLEX"

    Returns:
    float: The total cost of transportation (total variable costs) if an optimal solution is found, None otherwise
    """


    # Balance supply and demand
    supply, demand, costs, artificial_costs = get_balanced(supply, demand, costs)
    
    
    # Indices for supply and demand
    supply_nodes = range(len(supply))
    demand_nodes = range(len(demand))

    # Create problem
    problem = pulp.LpProblem("Transport_Problem", pulp.LpMinimize)

    # Decision variables
    # x[i][j]: Quantity, shipped from facility i to customer j
    x = pulp.LpVariable.dicts("shipment",
                              ((i, j) for i in supply_nodes for j in demand_nodes),
                              lowBound=0,
                              cat='Continuous')

    # Objective function: Minimize total transportation costs
    problem += pulp.lpSum(costs[i][j] * x[i, j] for i in supply_nodes for j in demand_nodes), "Total Transport Cost"

    # Supply constraints
    for i in supply_nodes:
        problem += pulp.lpSum(x[i, j] for j in demand_nodes) <= supply[i], f"Supply Constraint {i}"

    # Demand constraints
    for j in demand_nodes:
        problem += pulp.lpSum(x[i, j] for i in supply_nodes) >= demand[j], f"Demand Constraint {j}"
    


    # Solver selection based on the argument solver_type
    if solver_type == "CPLEX":
        problem.solve(pulp.CPLEX_CMD(msg=False))
    else:
        problem.solve(pulp.PULP_CBC_CMD(msg=False))



    # Check the status of the solution
    if problem.status == pulp.LpStatusOptimal:
        total_costs = pulp.value(problem.objective)
        total_costs = total_costs - artificial_costs
        return total_costs # Return the optimal total variable costs (transportation costs)
    
    else:        
        return None  # No optimal solution found





def varAnd(population, toolbox, cxpb, mutpb):
    
    """
    Unmodified version of the varAnd function from the DEAP framework, which is used to
    implement the variation of the population through crossover and mutation. 
    """
    
    
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """


    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring





# Overwrite the eaSimple from the DEAP framework with an adapted version
def eaSimpleAdapted(population, toolbox, cxpb, mutpb, ngen, meta_penalty, stats=None, halloffame=None, verbose=__debug__):
    
    """
    Adapted version of the eaSimple from the DEAP framework to effectively handle parallelization and the deployment of penalties
    
    This function, which is an adapted version of the eaSimple function from the DEAP framework, implements the main loop of an
    Genetic Algorithm. Since the NSGA-II was registered in the toolbox as selection operator, this main-loop uses a multi-objective
    optimization strategy that aims to minimze total costs while concurrently maximize demand coverage.
    
    The primary modificatons are the following:

    1.Parallelization of the fitness function
    Since the 'map' function from Python's 'multiprocessing' module instead of Python's standard 'map' function was registered in
    the toolobox, the individuals are evaluated by the fitness function not sequentially, but in parallel.
    
    2.Deployment of penalties through additional parameter 'meta_penalty'
    An additional parameter 'meta_penalty' was introduced, to enable the penalization of individuals 
    with insufficient demand coverage.
   
    3.Additional return value 'allBestInds'
    In the context of the objective to minimize total costs while concurrently maximizing demand coverage, the best individual
    does not orrespond to the individual where total costs and demand coverage have the best ratio, but rather to the individual 
    that, with a demand coverage of 100% or more, exhibits the minimal total costs. Therefore, a logic has been introduced that 
    identifies and stores the best individual according to the previous definition, found across the generations, and returns it
    in the variable 'allBestInds'.
    
   
    Additional Parameter
    meta_penalty : float
        Penalty value to penalize and therefore effectively exclude individuals with insufficient demand coverage
        from the optimization process.

    Additional Return value:
        allBestInds : list of deap.creator.Individual
        A list of best individuals from each generation.
    """
    


    """
    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    pareto_fronts = []
    allBestInds = []
    
    
    # Evaluate the individuals with an unevaluated (here called 'invalide', which means, that it due
    # to aforegoing modifications through crossover or mutation, is not exisiting) fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        
    pop_for_stats = [ind for ind in population if ind.fitness.values[0] != meta_penalty and ind.fitness.values[1] != 0]
        
    # Filter individuals with a fitness value for objective 1 of 100 or greater
    filtered_inds = [ind for ind in pop_for_stats if ind.fitness.values[1] >= 100]
    
    if halloffame is not None:
        halloffame.update(filtered_inds)
    
    if filtered_inds:
        # Find the individual with the minimal fitness value for objective 0 among the filtered individuals
        best_ind = min(filtered_inds, key=lambda ind: ind.fitness.values[0])
        
        # Store the found individual in the list for later analysis
        allBestInds.append(best_ind)
    
    
    record = stats.compile(filtered_inds) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
       print(logbook.stream)

    # Begin the generational process (main loop of the Genetic Algorithm):
    for gen in range(1, ngen + 1):
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an unevaluated (here called 'invalide', which means, that it due
        # to aforegoing modifications through crossover or mutation, is not exisiting) fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Replace the current population by the offspring
        population[:] = offspring

        pop_for_stats = [ind for ind in population if ind.fitness.values[0] != meta_penalty and ind.fitness.values[1] != 0]
        
   

        # Filter individuals with a fitness value for objective 1 of 100 or greater
        filtered_inds = [ind for ind in pop_for_stats if ind.fitness.values[1] >= 100]
        
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(filtered_inds)

        if filtered_inds:
            # Find the individual with the minimal fitness value for objective 0
            best_ind = min(filtered_inds, key=lambda ind: ind.fitness.values[0])
        
        # Store the found individual in the list for later analysis
        allBestInds.append(best_ind)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(filtered_inds) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)


        if verbose:
            print(logbook.stream)
    
    print("type of allBestInds:",type(allBestInds))

    return population, logbook, allBestInds





# Definition of the fitness function 'facilityFitness'
def facilityFitness(individual, cost_matrix, list_supply, list_demand, list_fixcosts, meta_penalty, cut_off, solver_type):
     
    """ 
    Fitness Function to Evaluate the fitness of an individual within the Genetic Algorithm
    
    This Function evaluates the fitness of an individual within the Genetic Algorithm, which effectively consists
    in calculating the total costs that involve the total fixed costs for all open facilities as well as the total
    transportation costs (total variable costs) for all necessary transports. Thereby, the transportation costs are
    calculated by the function transportation_method using linear programming to solve the classical transportation
    problem. Furthermore, this function applies penalties for individuals with insufficient demand coverage below a
    certain threshold called 'cut_off'.

    Parameters:
    individual : list of int
        Binary representation of a solution, where the bits of this binary string represent
        the facility states (0 for closed, 1 for open).
    cost_matrix : dict of numpy arrays
        Cost matrix where each element corresponds to the shipping costs per unit (float)
        from warehouse i (warehouse_index:row index) to customer j (customer_index: column index)
    list_supply : list of float
        List of supply quantities for each facility.
    list_demand : list of float
        List of demand quantities for each customer.
    list_fixcosts : list of float
        List of fixed costs for each warehouse.
    meta_penalty : float
        Penalty value to penalize and therefore effectively exclude individuals with insufficient demand coverage
        from the optimization process.
    cut_off : float
        Cut-off value for the demand coverage value in percent which is tolerated, below this value a individual
        is penalized with penalty costs excluding it effectively from the optimization process, default is 97.
    solver_type : str
        The solver type to be used for the transportation method ('CPLEX' or 'CBC').

    Returns:
    tuple
        total_costs : float
            The total cost including fixed and variable costs.
        demand_coverage : float
            The percentage of demand covered by the open facilities.

    Raises:
    Exception
        If the total supply is less than the total demand and penalty_costs is not provided.
    """


    # Activate the following comment to print process ids to check whether parallelization is done
    # print(f"Processing individual {individual[:5]}... on process ID: {os.getpid()}")


    # Convert cost_matrix from dictionary to 2D Numpy array if necessary
    if isinstance(cost_matrix, dict):
        cost_matrix = np.array([cost_matrix[key] for key in sorted(cost_matrix.keys())])
    
    # Filter out closed facilities based on the individual
    open_warehouse_indices = [i for i, status in enumerate(individual) if status == 1]
    
    # Handle the case of no open facilities: Return high penalty values
    if not open_warehouse_indices:  
        return meta_penalty, 0
    
    # Filter supply, demand, and costs for open facilities
    filtered_supply = [list_supply[i] for i in open_warehouse_indices]
    filtered_costs = cost_matrix[open_warehouse_indices, :]
    
    



    # Calculate demand coverage as a percentage
    total_supply = sum(filtered_supply)
    total_demand = sum(list_demand)
    
    if total_demand <= total_supply:
        demand_coverage = 100
        
    else:
        demand_coverage = (total_supply / total_demand) * 100 if total_demand > 0 else 0
    
    # Check for demand coverage less than cut off value (in percent), and return high cost to ensure these are dominated
    if demand_coverage < cut_off or demand_coverage > 100:
        return meta_penalty, 0
    
    
    
    
    
    # Calculate the total variable costs using the transportation method
    total_var_costs = transportation_method(filtered_supply, list_demand, filtered_costs, solver_type=solver_type)
    
    
    # Calculate the total number of open facilities
    open_warehouses = len(open_warehouse_indices)
  
    # Calculate total fixed costs for the open facilities
    total_fixed_costs = sum([list_fixcosts[i] for i in open_warehouse_indices])
    
    # Calculate total costs (Fitness)
    total_costs = total_var_costs + total_fixed_costs
    
    
    # Return the total costs and demand coverage
    return total_costs, demand_coverage





def initialize_parameters(warehouses):
    
    """
    Initializes the parameters for the genetic algorithm with default values.
    
    Parameters:
    warehouses : int
        The number of warehouses which determines the individual length.
    
    Returns:
    tuple
        IND_LENGTH : int
            The length of the individual (number of bits).
        POP_SIZE : int
            The size of the population.
        CX_PB : float
            The crossover probability.
        MUT_PB : float
            The mutation probability.
        MAX_GEN : int
            The maximum number of generations.
    """
    

    # Initialize problem parameters
    IND_LENGTH = warehouses  # String length in bits
    
    # Initialize the parameters of the genetic algorithm
    POP_SIZE = 100  # Number of individuals
    CX_PB = 0.9  # Crossover probability
    MUT_PB = 0.03  # Mutation probability
    MAX_GEN = 100  # Number of generations
    
    return IND_LENGTH, POP_SIZE, CX_PB, MUT_PB, MAX_GEN





def initialize_toolbox(IND_LENGTH, cost_matrix, list_supply, list_demand, list_fixcosts, meta_penalty, cut_off, separate_rng):
    
    """
    Initializes the toolbox for the genetic algorithm and registers genetic operators as well as the required functions
    such as the fitness function. This function also includes the setup of the multiprocessing pool and the registration
    of the corresponding map function, which implements the parallelization in the main loop of the Gentic Algorithm. 
    Prior to this, within this function, the utilized solver type (CPLEX or CBC) and the number of parallel processes 
    are determined based on the available resources.

    Parameters:
    IND_LENGTH : int
        The length of the individual (number of bits).
    cost_matrix : dict of numpy arrays
        Cost matrix where each element corresponds to the shipping costs per unit (float)
        from warehouse i (warehouse_index:row index) to customer j (customer_index: column index)
    list_supply : list of float
        List of capacities for each warehouse.
    list_demand : list of float
        List of demands for each customer.
    list_fixcosts : list of float
        List of fixed costs for each warehouse.
    meta_penalty : float
        Penalty value to penalize and therefore effectively exclude individuals with insufficient demand coverage
        from the optimization process.
    cut_off : float, optional
        Cut-off value for the demand coverage value in percent which is tolerated, below this value a individual
        is penalized with penalty costs excluding it effectively from the optimization process, default is 97.
    separate_rng : random.Random
        A separate random number generator instance for population creation.
    
    Returns:
    toolbox : deap.base.Toolbox
        The initialized DEAP toolbox with registered genetic operators and required functions.
    """


    # Select solver based on availability and print the selected solver
    if pulp.CPLEX_PY().available():
        solver = pulp.CPLEX_PY(msg=False)
        solver_type = "CPLEX"
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)
        solver_type = "CBC"

    print(f"\nUsing {solver_type} solver.\n")

   
    # Definition of a bi-objective fitness strategy
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
    
    # Definition of a list-based individual class
    creator.create("Individual", list, fitness=creator.Fitness)
    
    toolbox = base.Toolbox()

    # Determine the number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Number of available CPU cores: {num_cores}\n")


    # Parallelization: Creating a pool of processes with python's multiprocessing module
    if num_cores <= 59:
        # Set number of processes to 60 if 59 or less CPU cores are available
        pool = multiprocessing.Pool(processes=num_cores)

    else:
        # Set number of processes to 60 if more than 59 CPU cores are available
        pool = multiprocessing.Pool(processes=60)   
    
    # Toolbox-Registration of the created processing pool
    toolbox.register("map", pool.map)
    print(f"Number of parallel processes used: {pool._processes}\n\n\n")
   
    
    # Create and register an operator that randomly returns 0 or 1
    # Use separate random number generator instance 'rng' for population creation
    toolbox.register("zeroOrOne", separate_rng.randint, 0, 1)
    
    # Create and register an individual operator to fill the individual instances
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, IND_LENGTH)
    
    # Create and register the population operator that generates a list of individuals
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
    
    # Register the Fitness function
    toolbox.register("evaluate", facilityFitness, cost_matrix=cost_matrix, list_supply=list_supply, list_demand=list_demand, list_fixcosts=list_fixcosts, meta_penalty=meta_penalty, cut_off=cut_off, solver_type=solver_type)
    
    # Register the NSGA-II function as selection operator
    toolbox.register("select", tools.selNSGA2)
    
    # Register the One-Point-Crossover function
    toolbox.register("mate", tools.cxOnePoint)
    
    # Register the flip-bit mutation function: indpb is the probability for a bit to be flipped
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/IND_LENGTH)
    
    return toolbox





def run_genetic_algorithm(dataset, fixcosts_value, capacity=None, pop_seed=4, global_seed=15, cut_off=97, custom_paramslist = None):

    """
    Runs the genetic algorithm for a specified dataset and fixed costs.

    This function initializes the necessary parameters, sets up the genetic algorithm toolbox, and calls the function
    that executes the main loop of the genetic algorithm. It handles both default and custom parameter settings.

    Parameters:
    dataset : str
        The selected dataset identifier.
    fixcosts_value : str
        The selected fixed costs per facility.
    capacity : int, optional
        The selected capacity for datasets A, B, or C, default is None.
    pop_seed : int, optional
        The seed for the random number generator used in population creation, default is 4.
    global_seed : int, optional
        The global seed for random number generator of the gentic process involving crossover and mutation, default is 15.
    cut_off : float, optional
        Cut-off value for the demand coverage value in percent which is tolerated, below this value a individual
        is penalized with penalty costs excluding it effectively from the optimization process, default is 97.
    custom_paramslist : list, optional
        List of custom parameters [max_gen, pop_size, cx_pb, mut_pb, cut_off] to override defaults, default is None.
        If custom_paramslist is None, the default values for the parameters are max_gen=100, pop_size=100, cx_pb=0.9, 
        mut_pb=0.03, cut_off=97.

    Returns:
    tuple
        logbook : deap.tools.Logbook
            The logbook recording statistics for each generation.
        population : list of deap.creator.Individual
            The final population after the genetic algorithm run.
        hof : deap.tools.HallOfFame
            The hall of fame containing the best individual.
        bestInds1 : list of deap.creator.Individual
            List of best individuals from the genetic algorithm run.
    """


    warehouses, customers, list_supply, list_fixcosts, cost_matrix, list_demand, meta_penalty = read_data_create_matrix(dataset, fixcosts_value, capacity)

    # Create a separate instance of the random number generator for population creation
    pop_seed = pop_seed
    separate_rng = random.Random(pop_seed)
    
    # Set the global random number generator for reproducible results
    global_seed = global_seed
    random.seed(global_seed)

    
    if not custom_paramslist:
       # Initialize genetic parameters
       IND_LENGTH, POP_SIZE, CX_PB, MUT_PB, MAX_GEN = initialize_parameters(warehouses)
    
       # Intitialize the cut_off value
       cut_off = cut_off

    else: 
        # Assign parameters from custom_paramslist
        IND_LENGTH = warehouses  # String length in bits
        MAX_GEN, POP_SIZE, CX_PB, MUT_PB, cut_off = custom_paramslist
    
    print(f"\nSettings of the current optimization process: max_gen={MAX_GEN}, pop_size={POP_SIZE}, cxpb={CX_PB}, mutpb={MUT_PB}, cut_off={cut_off}")


    # Initialize the toolbox
    toolbox = initialize_toolbox(IND_LENGTH, cost_matrix, list_supply, list_demand, list_fixcosts, meta_penalty, cut_off, separate_rng)
    
    # Generate initial population, evaluate fitness and update history
    population = toolbox.populationCreator(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    
    # Initialize statistics for tracking performance across generations
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    
    # Execution of the Main loop for the Genetic Algorithm
    population, logbook, bestInds1 = eaSimpleAdapted(population, toolbox, cxpb=CX_PB, mutpb=MUT_PB, ngen=MAX_GEN, meta_penalty=meta_penalty, stats=stats, halloffame=hof, verbose=True)

    # Select best individual from final population to display its genotyp and its fitness
    print("\nBest individual represented as binary:", hof[0])
    print("\nWith the fitness:", hof[0].fitness)

    return logbook, population, hof, bestInds1





def get_combinations():
    
    """
    Generates all possible valid combinations of dataset identifier and fixcost value
    for automatic execution of all datasets from IV to XIII.
    
    Returns:
    list of tuple
        A list of tuples, each containing a dataset identifier and a fixed cost value.
    """
    

    datasets = ['IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII']
    fixcosts_values = [7500, 12500, 17500, 25000]
    
    # Define combinations to be excluded
    excluded_combinations = [('V', '7500'), ('V', '12500'), ('V', '25000')]
    
    # Convert excluded_combinations to include string representation of fixcosts for consistency
    excluded_combinations = [(ds, str(fc)) for ds, fc in excluded_combinations]
    
    # Generate combinations, excluding the specified ones
    combinations = [(dataset, str(fixcost)) for dataset in datasets for fixcost in fixcosts_values
                    if (dataset, str(fixcost)) not in excluded_combinations]
    
    return combinations










def main():

    """
    Main entry point for the execution of the genetic algorithm based on user-selected mode.

    This function effectively composes all the functions above to one working Hybride Approach which uses a parallelized fitness
    function and has the capability to solve the the test instances for the Capacitated Facility Location Problem (CFLP) from the
    OR Library, presented by Beasley in 1988. Therefore, the function initially prompts the user to choose between running the 
    genetic algorithm for a single dataset, automatically for the datasets 'IV' to 'XIII', or automatically for the large sets 'A',
    'B' and 'C'. Depending on the user's choice, it either prompts for values to select an available dataset and asks for customization
    of the Algorithm's parameters or only asks for customization of the Algorithm's parameters. After running the algorithm, execution
    time and further relevant informations for each run, which were captured during the execution are printed (in single mode) or 
    stored in an excel file on the computer.
  
    Returns:
    tuple
        If meta_choice is 'single':
            logbook : deap.tools.Logbook
                The logbook recording statistics for each generation.
            population : list of deap.creator.Individual
                The final population after the genetic algorithm run.
            hof : deap.tools.HallOfFame
                The hall of fame containing the best individual.
            bestInds1 : list of deap.creator.Individual
                A list of best individuals from each generation.

        If meta_choice is 'auto' or 'autolarge':
            logbook : deap.tools.Logbook
                The logbook recording statistics for each generation.
            population : list of deap.creator.Individual
                The final population after the genetic algorithm run.
            hof : deap.tools.HallOfFame
                The hall of fame containing the best individual.
            bestInds1 : list of deap.creator.Individual
                A list of best individuals from each generation.
            all_hofs : dict
                A dictionary storing Hall of Fame objects for each dataset and fixcost combination.
            execution_times : list of float
                A list of execution times for each run.
    """


    meta_choice = execution_mode()
    
    if meta_choice == 'single':
        dataset, fixcosts_value = user_query()  # Obtain user input for dataset and fix costs
        custom_paramslist = customize_params() # ask user to customize parameters or use default values
        start_time = time.time()  # Capture start time
        logbook, population, hof, bestInds1 = run_genetic_algorithm(dataset, fixcosts_value, custom_paramslist=custom_paramslist)
        end_time = time.time()  # Capture end time
        
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time for the algorithm: {execution_time:.4f} seconds")  # Print execution time

        return logbook, population, hof, bestInds1
        
    elif meta_choice == 'auto':
        combinations = get_combinations()
        all_hofs = {}  # Initialize a collection to store Hall of Fame objects for each execution
        execution_times = []  # Initialize a list to store execution times
        custom_paramslist = customize_params() # ask user to customize parameters or use default values
        
        for dataset, fixcosts_value in combinations:
            print(f"\nRunning for dataset {dataset} with fixcosts {fixcosts_value}")
            start_time = time.time()  # Capture start time
            logbook, population, hof, bestInds1 = run_genetic_algorithm(dataset, fixcosts_value, custom_paramslist=custom_paramslist)
            end_time = time.time()  # Capture end time
            execution_time = end_time - start_time  # Calculate execution time
            execution_times.append(execution_time)
            print(f"Execution time for the algorithm: {execution_time:.4f} seconds")  # Print execution time
            all_hofs[(dataset, fixcosts_value)] = hof  # Store each hof
            
        # Return `all_hofs` for auto mode indicating multiple runs
        return logbook, population, hof, bestInds1, all_hofs, execution_times
    
    elif meta_choice == 'autolarge':
        datasets = ['a', 'b', 'c']
        capacities = {
            'a': [8000, 10000, 12000, 14000],
            'b': [5000, 6000, 7000, 8000],
            'c': [5000, 5750, 6500, 7250]
        }
        pop_seed_values = [4, 4, 4, 4, 4, 4, 95, 95, 95, 95, 95, 95]
        global_seed_values = [82, 82, 82, 15, 15, 15, 82, 82, 82, 15, 15, 15]
        cut_off_values = [90, 95, 97, 90, 95, 97, 90, 95, 97, 90, 95, 97]
        
        all_hofs = {}  # Initialize a collection to store Hall of Fame objects for each execution
        execution_times = []  # Initialize a list to store execution times
        custom_paramslist = customize_params() # ask user to customize parameters or use default values
        
        for dataset in datasets:
            for capacity in capacities[dataset]:
                for i in range(len(pop_seed_values)):
                    pop_seed = pop_seed_values[i]
                    global_seed = global_seed_values[i]
                    cut_off = cut_off_values[i]
                    
                    print(f"\nRunning for dataset {dataset} with capacity {capacity}, pop_seed {pop_seed}, global_seed {global_seed}, cut_off {cut_off}")
                    start_time = time.time()  # Capture start time
                    logbook, population, hof, bestInds1 = run_genetic_algorithm(dataset, None, capacity, pop_seed, global_seed, cut_off, custom_paramslist=custom_paramslist)
                    end_time = time.time()  # Capture end time
                    execution_time = end_time - start_time  # Calculate execution time
                    execution_times.append(execution_time)
                    print(f"Execution time for the algorithm: {execution_time:.4f} seconds")  # Print execution time
                    all_hofs[(dataset, capacity, pop_seed, global_seed, cut_off)] = hof  # Store each hof
            
        # Return `all_hofs` for autolarge mode indicating multiple runs
        return logbook, population, hof, bestInds1, all_hofs, execution_times






# Execution and Handling of the Results
if __name__ == "__main__":
    results = main()

    # Determine if results include `all_hofs` based on length
    if len(results) == 6:
        # Auto mode: all_hofs is included
        logbook, population, hof, bestInds1, all_hofs, execution_times = results
        print("All Execution Times:\n\n")
        print(execution_times)
        
        # Initialize lists to collect fitness and individual data
        fitness_data = []
        individuals_data = []

        # Print all dataset and fixcost keys
        for key in all_hofs.keys():
            print(f"Dataset and Fixcosts: {key}, Best Individual: {all_hofs[key][0]}")
            
            if all_hofs[key]:  # Ensure there is at least one individual per key
                individual = all_hofs[key][0]  # Get the first individual for the key
                fitness_values = individual.fitness.values
                key_str = (str(key[0]), str(key[1]))  # Convert key parts to strings
                index_str = '0'  # Since there is only one individual per key, index is 0
                # Append fitness data, converting all numbers to strings
                fitness_data.append([*key_str, index_str] + [str(val) for val in fitness_values])
                # Append individuals data, converting all genes to strings
                individuals_data.append([*key_str, index_str] + [str(gene) for gene in individual])

        # Define the columns for each DataFrame
        fitness_columns = ['Testset', 'Fixcosts', 'Index', 'Fitness_1\nTotal Costs', 'Fitness_2\nSupply Coverage']
        individuals_columns = ['Testset', 'Fixcosts', 'Index'] + [f'Gene_{i+1}' for i in range(len(individual))]

        # Create DataFrames
        fitness_df = pd.DataFrame(fitness_data, columns=fitness_columns)
        individuals_df = pd.DataFrame(individuals_data, columns=individuals_columns)

        # Print all individuals and fitness values before saving
        print("Individuals Data:")
        print(individuals_df)
        print("\nFitness Values Data:")
        print(fitness_df)

        # Base paths for saving the Excel files (can be modified by the user as needed)
        base_path_fitness = os.getcwd() + '/Fitnessvalues/'
        base_path_individuals = os.getcwd() + '/Individuals/'

        # Ensure directories exist
        os.makedirs(base_path_fitness, exist_ok=True)
        os.makedirs(base_path_individuals, exist_ok=True)

        # Save to Excel
        fitness_df.to_excel(base_path_fitness + 'fitness_values_new111.xlsx', index=False)
        individuals_df.to_excel(base_path_individuals + 'individuals_new111.xlsx', index=False)

        print("Excel files have been saved successfully.")
        
    else:
        # Single mode: Only standard results are returned
        logbook, population, hof, bestInds1 = results






