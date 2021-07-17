# coding=utf-8

import time
import argparse

from multiprocessing import Pool, cpu_count

from itertools import product
from math import factorial
from functools import partial

import matplotlib.pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def create_random_solution(n, m):
    '''
    Creates a random solution given the restrictions for n and m.
    Given n = 6 and m = 3, for example, return a random solution
    with 6 numbers out of which 3 must be one, and the rest 0.
    '''
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # aquellos índices que superen un umbral se ponen a 1
    M[np.random.rand(n) > 0.3] = 1

    M = shape_solution(M, m)
    return M


def read_distance_matrix(path):
    '''
    Helper function to read the upper triangluar of the 
    distance matrix from file.
    Returns n, m and the values themselves.
    '''
    with open(path) as file:
        lines = file.readlines()


    print("Reading {} lines".format(len(lines) - 1))
    n, m = lines[0].split(" ")
    n, m = int(n), int(m)
    data = np.array([float(line.strip().split(" ")[2]) for line in lines[1:]])
    return n, m, data


def fill_upper_triangular(a):
    '''
    Creates and returns a (n, n) matrix with the upper triangular
    filled with the data recovered from the files.
    '''
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n, dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=float)
    out[mask] = a
    return out.T


def shape_solution(M, m):
    '''
    Helper function to ensure the solution given M is
    a feasible solution. That is, `sum(M) == m`. 

    If this does not apply, check what is the case 
    (`sum(M) < m` or `sum(M) > m`) and flip values so that
    the resulting array has ``sum(M) == m``.
    '''
    M = np.array(M)
    while np.sum(M) > m:
        ones = M.nonzero()[0]
        random_one = np.random.choice(ones)
        M[random_one] = 0

    while np.sum(M) < m:
        zeros = np.where(M==0)[0]
        random_zero = np.random.choice(zeros)
        M[random_zero] = 1

    return M

   
def calculate_diversity(M, D):
    '''
    This function calculates the diversity of a solution based on the
    definition for the Maximum Diversity Problem.

    Given the distance matrix, ``D`` and a solution `M`, calculate
    ``sum(D[i,j] * M[i] * M[j])`` for all ``i`` and `j` in `M`, corresponding
    to all the possible combinations ``(i, j)`` of values within `M`.

    This calculation is optimized to work over symmetric `D`s, that is, 
    distance matrices that have only the upper triangular filled with data.
    '''

    # first, find all the possible combination of indices that lie
    # within the upper triangular section of our n x n matrix
    indices_triu = np.argwhere(np.triu(np.ones((len(M),) * 2),1))
    indices_triu = indices_triu[indices_triu[:,0] < indices_triu[:,1]]

    # second, find all posible combinations of genotypes
    # within our particular solution
    mesh = np.array(np.meshgrid(M, M))
    combs = mesh.T.reshape(-1, 2)

    # third, calculate all the possible combinations of genotypes, 
    # just as before, but now with the indices, not the values themselves
    
    M_i = np.indices((len(M),))
    mesh_i = np.meshgrid(M_i, M_i)
    combs_i = np.transpose(mesh_i).reshape(-1, 2)

    # given those, find all combinations that match our
    # upper triangular section rule, just like before
    combs_triu = combs[combs_i[:,0] < combs_i[:,1]]

    # stack the results so we have rows in the format: [gen1, gen2, gen1_index, gen2_index]
    col_stack = np.column_stack((combs_triu, indices_triu))

    # To calculate the diversity, access the distances matrix with
    # gen1_index and gen2_index, and multiply by the values gen1 and gen2 themselves.
    # If any of the values is 0, this will all be canceled out. 
    # This will return a vector that will contain either the value 
    # of the distance between two particular genes, or 0. Sum it all up and return.
    return np.sum([D[ i[2], i[3] ] * i[0] * i[1] 
                    for i in col_stack])


def brute_force(n, D, m):
    '''
    Implementation for the brute force version to find a solution for this problem.
    Useful to test performance over very small problems (n < 20)
    '''
    # extract all posible combinations of m cardinality to
    # reduce search space
    all_combs = [np.array(i) 
                for i in product([0, 1], repeat=n)]

    filtered_combs = list(filter(lambda x: np.sum(x)==m, all_combs))

    print("Espacio de posibles soluciones para {} elementos y m = {}: {}\n".format(n, m, len(filtered_combs)))

    all_solutions = [calculate_diversity(c, D) for c in filtered_combs]

    index = np.argmax(all_solutions)
    max_v = all_solutions[index]
    best_comb = filtered_combs[index]
    print("La mejor solución es {}, en el índice {} con una diversidad de {}".format(best_comb, index, max_v))


def mutation(M, m, m_factor):
    '''
    Mutate a solution. For each genotype within the solution, flip it with some m_factor probability.
    '''
    return shape_solution([genotype if np.random.rand() > m_factor else 1 - genotype for genotype in M], m)




def classic_crossover(father, mother, m):
    '''
    Implementation of the classic crossover between two solutions.

    Find a cross point, and copy the left part of the father and the
    right part of the mother to the new solution.

    The inverse is also considered, returning two possible solutions. These
    are then shaped to match our problem's criteria.
    '''

    # take random point
    rand_index = np.random.randint(1, len(father) - 1)
    # copy first half of father to child
    lh_f = father[0:rand_index] # left hand father
    rh_f = father[rand_index:] # right hand father
    # copy last half of mother to child
    lh_m = mother[0:rand_index] # left hand mother
    rh_m = mother[rand_index:] # right hand mother
 
    # creation of the new generation
    child_1 = np.concatenate((lh_f, rh_m))
    child_2 = np.concatenate((lh_m, rh_f))

    # make sure the solutions are valid
    child_1 = shape_solution(child_1, m)
    child_2 = shape_solution(child_2, m)
    return (child_1, child_2)



def two_point_crossover(father, mother, m):
    '''
    Implementataion of the two point crossover. Find two crossover points
    and copy the inside of the father and the outsides of the mother to 
    the new solution.

    The inverse is also considered, returning two possible solutions. These
    are then shaped to match our problem's criteria. 
    '''


    point_1 = np.random.randint(1, len(father) // 2)
    point_2 = np.random.randint(point_1 + 1, len(father) - 1)

    inside_f = father[point_1:point_2]
    inside_m = mother[point_1:point_2]

    outside_l_f = father[0:point_1]
    outside_r_f = father[point_2:]

    outside_l_m = mother[0:point_1]
    outside_r_m = mother[point_2:]

    child_1 = np.concatenate((outside_l_f, inside_m, outside_r_f))
    child_2 = np.concatenate((outside_l_m, inside_f, outside_r_m))
    
    child_1 = shape_solution(child_1, m)
    child_2 = shape_solution(child_2, m)

    return (child_1, child_2)


def genetic_algorithm(n, m, D, initial_population=100, k_top=15, m_factor=0.002, n_iterations=500, patience=20):
    '''
    Implementation of the genetic algorithm. Given n, m, D, find a solution to the problem
    using a genetic approach.

    ``initial_population`` defines how many solutions will be generated and how big will each
    generation be.

    ``k_top`` defines how many solutions are kept to reproduce.

    ``m_factor`` defines de mutation factor for each genotype in each solution.

    ``n_iterations`` defines a maximum number of iterations.

    ``patience`` defines a patience counter. This counter will decrease if the best solution
    found so far has not changed. If the counter runs out, assume stabilization of the 
    algoritm and return the best solution found. If a new best solution is found, the patience counter is reset to 0.
    '''

    if initial_population < k_top:
        raise ValueError("initial_population must be greater than k_top")

    # Initialization of misc variables to help in the calculation
    pool = Pool(cpu_count())
    partial_func = partial(calculate_diversity, D=D)

    # Initialize population
    # For this, lets generate 1 possible solution and calculate permutations over it
    current_generation = [create_random_solution(n, m) for i in range(initial_population)]
    diversity_arr = pool.map(partial_func, current_generation)

    # To keep track during the execution
    current_best_solution_d = 0
    last_best_solution = 0
    counter = 0

    # variables for plotting data
    best_solution_history = []
    patience_history = []
    best_solution_gen_history = []


    for i in range(n_iterations):
        print("\n\n*** Iteration {} ***\n\n".format(i))

        # sort this generation based on fitness
        gen_div = list(zip(current_generation, diversity_arr)) 
        sorted_gen_div = sorted(gen_div, key = lambda x: x[1], reverse=True)

        # Select k_top best parents from this generation
        best_solutions = [s[0] for s in sorted_gen_div]
        survivals = best_solutions[:k_top]

        print("> Best solution in gen {} had diversity {}\n".format(i, sorted_gen_div[0][1]))
        
        # given the survivals, generate all possible combination of 
        # pairs between them
        pairs = np.squeeze(sliding_window_view(survivals, (2, n)))

        # apply crossover for all pairs to build next generation
        current_generation = [two_point_crossover(pair[0], pair[1], m) for pair in pairs]
        current_generation = np.reshape(current_generation, (2 * (k_top - 1), n))

        # Mutation
        current_generation = [mutation(solution, m, m_factor) for solution in current_generation]
        
        # calculate fitness for this generation
        diversity_arr = pool.map(partial_func, current_generation)


        
        # for plotting
        best_solution_gen_history.append(sorted_gen_div[0][1])

        if current_best_solution_d < sorted_gen_div[0][1]:
            current_best_solution_d = sorted_gen_div[0][1]
            current_best_solution = sorted_gen_div[0][0]
            counter = 0

        if(last_best_solution == current_best_solution_d):
            counter += 1
        else:
            last_best_solution = current_best_solution_d

        print("> Best solution so far has diversity {}\n".format(last_best_solution))
        
        # for plotting
        best_solution_history.append(last_best_solution)
        patience_history.append(counter)

        print("> Patience counter: {}. {} more to finish if equal.".format(counter, patience - counter))
        if counter == patience:
            print("\nPatience counter dropped to 0.\n")
            print("\nBest solution found had diversity {}".format(current_best_solution_d))
            print("\nBest solution was {}".format(current_best_solution))
            return (current_best_solution, current_best_solution_d), (best_solution_gen_history, best_solution_history, patience_history)


    print("\nRan out of iterations.\n")
    print("\nBest solution found had diversity {}".format(current_best_solution_d))
    print("\nBest solution was {}".format(current_best_solution))

    return (current_best_solution, current_best_solution_d), (best_solution_gen_history, best_solution_history, patience_history)

    




if __name__ == "__main__":
    np.random.seed(7)

    argparse = argparse.ArgumentParser("Pass in the file to process")
    argparse.add_argument('--file', '-f', type=str)

    args = argparse.parse_args()

    if args.file is not None:

        n, m, data = read_distance_matrix("data/{}".format(args.file))
        print(data)
        D = fill_upper_triangular(data)
    else:
        n, m = 300, 50
        D = np.random.randint(100, size=(n, n), dtype=int)


    print("Distance matrix:\n")
    print(D)
    print()
    print()


    start_time = time.time()
    solution, historic_data = genetic_algorithm(n, m, D, 
                            initial_population=50, 
                            k_top=20,
                            m_factor=0.002, 
                            n_iterations=100, 
                            patience=20)
    end_time = time.time()


    total_space = factorial(n) // (factorial(n - m) * factorial(m))
    print("\nTotal time elapsed: {} seconds.".format(end_time - start_time))
    print("\nSearch space contains {} possible solutions for n = {} and m = {}.".format(total_space, n, m))

    best_solution_gen_history, best_solution_history, patience_history = historic_data


    plt.plot(list(range(len(best_solution_history))), best_solution_history, linewidth=3)
    plt.plot(list(range(len(best_solution_gen_history))), best_solution_gen_history, 'y--', linewidth=3)

    plt.ylabel("Diversity")
    plt.xlabel("Generation")
    
    if args.file is not None:
        plt.title(args.file[:-4])
        plt.tight_layout()
        plt.savefig("results/{}.pdf".format(args.file[:-4]))
    else:
        plt.tight_layout()
        plt.show()

