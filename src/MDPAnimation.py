from itertools import product
from manimlib import *
import numpy as np
from numpy.core.fromnumeric import sort
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(2)


# Distance Matrix
n = 8
m = 4
D = np.random.randint(100, size=(n, n), dtype=int)
np.fill_diagonal(D, 0)
D = np.triu(D)

# TITLE = Text("Maximum Diversity Problem", font="SF Mono")
# MDP_FORMULA = Tex(r"MD(x) = \sum^{n-1}_{i=0} \sum^{n}_{j=i+1} D_{ij}x_ix_j")


# MDP Functions
def create_random_solution(n, m):
    """
    Creates a random solution given the restrictions for n and m.
    Given n = 6 and m = 3, for example, return a random solution
    with 6 numbers out of which 3 must be one, and the rest 0.
    """
    # creamos una solución vacía
    M = np.zeros(n, dtype=int)

    # aquellos índices que superen un umbral se ponen a 1
    M[np.random.rand(n) > 0.3] = 1

    M = shape_solution(M, m)
    return M


def fill_upper_triangular(a):
    """
    Creates and returns a (n, n) matrix with the upper triangular
    filled with the data recovered from the files.
    """
    n = int(np.sqrt(len(a) * 2)) + 1
    mask = np.tri(n, dtype=bool, k=-1)  # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n, n), dtype=float)
    out[mask] = a
    return out.T


def shape_solution(M, m):
    """
    Helper function to ensure the solution given M is
    a feasible solution. That is, `sum(M) == m`.

    If this does not apply, check what is the case
    (`sum(M) < m` or `sum(M) > m`) and flip values so that
    the resulting array has ``sum(M) == m``.
    """
    M = np.array(M)
    while np.sum(M) > m:
        ones = M.nonzero()[0]
        random_one = np.random.choice(ones)
        M[random_one] = 0

    while np.sum(M) < m:
        zeros = np.where(M == 0)[0]
        random_zero = np.random.choice(zeros)
        M[random_zero] = 1

    return M


def calculate_diversity(M, D):
    """
    This function calculates the diversity of a solution based on the
    definition for the Maximum Diversity Problem.

    Given the distance matrix, ``D`` and a solution `M`, calculate
    ``sum(D[i,j] * M[i] * M[j])`` for all ``i`` and `j` in `M`, corresponding
    to all the possible combinations ``(i, j)`` of values within `M`.

    This calculation is optimized to work over symmetric `D`s, that is,
    distance matrices that have only the upper triangular filled with data.
    """

    # first, find all the possible combination of indices that lie
    # within the upper triangular section of our n x n matrix
    indices_triu = np.argwhere(np.triu(np.ones((len(M),) * 2), 1))
    indices_triu = indices_triu[indices_triu[:, 0] < indices_triu[:, 1]]

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
    combs_triu = combs[combs_i[:, 0] < combs_i[:, 1]]

    # stack the results so we have rows in the format: [gen1, gen2, gen1_index, gen2_index]
    col_stack = np.column_stack((combs_triu, indices_triu))

    # To calculate the diversity, access the distances matrix with
    # gen1_index and gen2_index, and multiply by the values gen1 and gen2 themselves.
    # If any of the values is 0, this will all be canceled out.
    # This will return a vector that will contain either the value
    # of the distance between two particular genes, or 0. Sum it all up and return.
    return np.sum([D[i[2], i[3]] * i[0] * i[1] for i in col_stack])


def brute_force(n, D, m):
    """
    Implementation for the brute force version to find a solution for this problem.
    Useful to test performance over very small problems (n < 20)
    """
    # extract all posible combinations of m cardinality to
    # reduce search space
    all_combs = [np.array(i) for i in product([0, 1], repeat=n)]

    filtered_combs = list(filter(lambda x: np.sum(x) == m, all_combs))

    print(
        "Espacio de posibles soluciones para {} elementos y m = {}: {}\n".format(
            n, m, len(filtered_combs)
        )
    )

    all_solutions = [calculate_diversity(c, D) for c in filtered_combs]

    index = np.argmax(all_solutions)
    max_v = all_solutions[index]
    best_comb = filtered_combs[index]
    print(
        "La mejor solución es {}, en el índice {} con una diversidad de {}".format(
            best_comb, index, max_v
        )
    )


def mutation(M, m, m_factor):
    """
    Mutate a solution. For each genotype within the solution, flip it with some m_factor probability.
    """
    return shape_solution(
        [genotype if np.random.rand() > m_factor else 1 - genotype for genotype in M], m
    )


def classic_crossover(father, mother, m):
    """
    Implementation of the classic crossover between two solutions.

    Find a cross point, and copy the left part of the father and the
    right part of the mother to the new solution.

    The inverse is also considered, returning two possible solutions. These
    are then shaped to match our problem's criteria.
    """

    # take random point
    rand_index = np.random.randint(1, len(father) - 1)
    # copy first half of father to child
    lh_f = father[0:rand_index]  # left hand father
    rh_f = father[rand_index:]  # right hand father
    # copy last half of mother to child
    lh_m = mother[0:rand_index]  # left hand mother
    rh_m = mother[rand_index:]  # right hand mother

    # creation of the new generation
    child_1 = np.concatenate((lh_f, rh_m))
    child_2 = np.concatenate((lh_m, rh_f))

    # make sure the solutions are valid
    child_1 = shape_solution(child_1, m)
    child_2 = shape_solution(child_2, m)
    return (child_1, child_2)


def two_point_crossover(father, mother, m):
    """
    Implementataion of the two point crossover. Find two crossover points
    and copy the inside of the father and the outsides of the mother to
    the new solution.

    The inverse is also considered, returning two possible solutions. These
    are then shaped to match our problem's criteria.
    """

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


# Manim Classes
class MMatrix:
    def __init__(self, D) -> None:
        numbers_to_text = [
            Text(f"{num}", font="SF Mono", color=YELLOW) for num in D.ravel()
        ]
        dist_matrix = VGroup(*numbers_to_text)
        dist_matrix.arrange_in_grid(n_rows=n, buff=0.4)
        self.mmatrix = dist_matrix

    def get(self) -> VGroup:
        return self.mmatrix


class Butterfly:
    def __init__(self, color=BLUE) -> None:
        self.svg = (
            SVGMobject("butterfly-top-view.svg")
            .set_stroke(width=0.01)
            .set_color(color)
            .rotate(-0.6)
            .scale(0.6)
        )

    def get(self) -> SVGMobject:
        return self.svg


class SolutionSquare(VGroup):
    def __init__(self, n):
        self.square = Square()
        self.number = Text(str(n), font="SF Mono")

        if n == 0:
            self.square.set_fill(BLACK, opacity=1).set_stroke(width=1)
            self.number.set_color(WHITE)

        else:
            self.square.set_fill(WHITE, opacity=1).set_stroke(width=1)
            self.number.set_color(BLACK)

        super().__init__(self.square, self.number)
        self.arrange(ORIGIN, buff=0)


class Solution(VGroup):
    def __init__(self, M):
        self.squares = VGroup()
        for i in range(len(M)):
            self.squares.add(SolutionSquare(M[i]))

        self.diversity = Text(
            str(calculate_diversity(M, D)), color=YELLOW, font="SF Mono"
        )

        super().__init__(self.squares, self.diversity)
        self.squares.arrange_in_grid(n_rows=1, buff=0.01)
        self.arrange(RIGHT, buff=1)


class Generation(VGroup):
    def __init__(self, generation: np.ndarray):

        self.gen_dict = {
            index: Solution(solution) for index, solution in enumerate(generation)
        }

        self.generation = VGroup(*self.gen_dict.values())
        self.generation.arrange_in_grid(n_cols=1, buff=1)
        super().__init__(self.generation)

    def __getitem__(self, value) -> VGroup:
        if isinstance(value, slice):
            return VGroup(*list(self.gen_dict.values())[value])
        else:
            return self.gen_dict[value]

    def delete_solution(self, index):
        pass

    def process_slice(self, section, function):
        return ApplyFunction(
            function, VGroup(*[i[1] for i in self.gen_dict.items()][section])
        )

    def swap_solutions(self, i1, i2):
        self.gen_dict[i1], self.gen_dict[i2] = self.gen_dict[i2], self.gen_dict[i1]


# Scenes
class IntroButterflies(Scene):
    def construct(self):

        intro_btf = Butterfly().get()

        self.play(FadeIn(intro_btf, RIGHT), run_time=1)
        self.wait(2)
        self.play(FadeOut(intro_btf, LEFT))

        butterflies = VGroup(
            *[
                Butterfly(random_bright_color())
                .get()
                .next_to(RIGHT * i, buff=0.8)
                .rotate(np.random.uniform(-0.6, 0.6))
                .scale(np.random.uniform(0.3, 0.9))
                for i in range(6)
            ]
        ).move_to(ORIGIN)

        self.play(LaggedStartMap(FadeIn, butterflies), run_time=3)
        self.wait(1)
        self.play(LaggedStartMap(FadeOut, butterflies), run_time=3)

        butterflies = (
            VGroup(
                *[
                    Butterfly(random_color())
                    .get()
                    .rotate(np.random.uniform(-0.6, 0.6))
                    .scale(np.random.uniform(0.3, 0.9))
                    for i in range(30)
                ]
            )
            .move_to(ORIGIN)
            .arrange_in_grid()
        )

        self.play(
            LaggedStartMap(FadeIn, butterflies),
            run_time=1,
        )

        self.wait(2)
        self.play(
            LaggedStartMap(FadeOut, butterflies),
            run_time=1,
        )

        self.wait(2)

        butterfly_measures = Butterfly(random_color()).get().scale(1.5)
        self.play(FadeInFromPoint(butterfly_measures, butterfly_measures.get_center()))

        self.wait()
        brace_height = Brace(butterfly_measures, LEFT)
        self.play(Write(brace_height))

        height_txt = Text("Height").next_to(brace_height, LEFT, buff=-0.4).scale(0.4)
        self.play(Write(height_txt))

        brace_width = Brace(butterfly_measures, UP)
        self.play(Write(brace_width))

        width_txt = Text("Width").next_to(brace_width, UP, buff=-0.05).scale(0.4)
        self.play(Write(width_txt))

        self.play(
            butterfly_measures.animate.set_color(random_color()),
            run_time=0.5,
        )

        self.wait(0.5)
        technical_info = (
            VGroup(
                *[
                    Text(t).scale(0.6)
                    for t in [
                        "Scientific Name",
                        "Family",
                        "Order",
                        "Life Cycle",
                        "Migration Route",
                        "...",
                    ]
                ]
            )
            .arrange_in_grid(n_cols=1)
            .move_to(RIGHT * 2)
        )

        group_scene = VGroup(
            butterfly_measures, brace_height, height_txt, brace_width, width_txt
        )
        self.play(group_scene.animate.shift(LEFT * 3))

        self.play(Write(technical_info))

        self.wait(2)
        group_scene.add(technical_info)
        self.play(FadeOut(group_scene))


class TestSolution(Scene):
    def construct(self):
        generation = [create_random_solution(n, m) for _ in range(10)]
        gen_repr = Generation(generation).scale(0.3)
        self.play(Write(gen_repr))

        self.play(Transform(gen_repr[4], gen_repr[4].set_opacity(0.2)))
        self.play(FadeOut(gen_repr[3]))


class GeneticAlgorithm(Scene):
    def construct(self):

        initial_population = 10
        k_top = 7
        m_factor = 0.002
        n_iterations = 500
        patience = 20

        current_generation = [
            create_random_solution(n, m) for _ in range(initial_population)
        ]

        current_generation_repr = Generation(current_generation).scale(0.4)

        self.play(LaggedStartMap(FadeIn, current_generation_repr))

        diversity_arr = [calculate_diversity(s, D) for s in current_generation]
        print(diversity_arr)

        self.play(
            current_generation_repr.animate.scale(0.5),
        )
        self.play(
            current_generation_repr.animate.shift(UP * 0.8 + LEFT * 4.3),
        )

        # To keep track during the execution
        current_best_solution_d = 0
        last_best_solution = 0
        counter = 0

        # variables for plotting data
        best_solution_history = []
        patience_history = []
        best_solution_gen_history = []

        gen_text = Text("Generation: ").move_to(UP * 3 + RIGHT * 3)
        gen_counter = Integer(0).next_to(gen_text, RIGHT).scale(0.5)
        gen_c_vg = VGroup(gen_text, gen_counter).scale(0.5)

        best_text = Text("Best solution so far: ").next_to(gen_text, DOWN)
        best_div = Integer(0).next_to(best_text, RIGHT).scale(0.5)
        best_vg = VGroup(best_text, best_div).scale(0.5)

        self.play(
            FadeIn(
                VGroup(gen_c_vg, best_vg).arrange(DOWN, center=False, aligned_edge=LEFT)
            )
        )

        for i in range(n_iterations):

            print("\n\n*** Iteration {} ***\n\n".format(i))

            # sort this generation based on fitness
            gen_div = list(zip(current_generation, diversity_arr))
            sorted_gen_div = sorted(gen_div, key=lambda x: x[1], reverse=True)

            sorted_gen_repr = (
                Generation([s[0] for s in sorted_gen_div])
                .next_to(current_generation_repr, RIGHT, buff=-4)
                .scale(0.2)
            )

            sorting_animation = []
            aux_sort_arr = diversity_arr.copy()
            for i in range(len(aux_sort_arr)):
                sorted_indices = np.argsort(aux_sort_arr)[::-1]
                print(sorted_indices)
                sorting_animation.append(
                    self.play(
                        Swap(
                            current_generation_repr[i],
                            current_generation_repr[sorted_indices[i]],
                        ),
                        run_time=10 / (i + 3),
                    )
                )
                current_generation_repr.swap_solutions(i, sorted_indices[i])
                aux_sort_arr[i], aux_sort_arr[sorted_indices[i]] = (
                    aux_sort_arr[sorted_indices[i]],
                    aux_sort_arr[i],
                )
                print("Aux sort:", aux_sort_arr)

            # self.play(*sorting_animation, run_time=10)

            self.wait(100)
            self.play(FadeOut(current_generation_repr), FadeIn(sorted_gen_repr))

            # Select k_top best parents from this generation
            best_solutions = [s[0] for s in sorted_gen_div]
            survivals = best_solutions[:k_top]

            survivals_repr = (
                Generation(survivals).move_to(UP * 0.8 + LEFT * 4.3).scale(0.2)
            ).next_to(sorted_gen_repr, ORIGIN, aligned_edge=UP)

            self.wait(2)

            def fade_section(mobject):
                return mobject.set_opacity(0.3)

            self.play(
                survivals_repr.process_slice(
                    slice(k_top + 1, len(survivals)), fade_section
                )
            )

            print(
                "> Best solution in gen {} had diversity {}\n".format(
                    i, sorted_gen_div[0][1]
                )
            )

            # given the survivals, generate all possible combination of
            # pairs between them
            pairs = np.squeeze(sliding_window_view(survivals, (2, n)))

            # apply crossover for all pairs to build next generation
            current_generation = [
                two_point_crossover(pair[0], pair[1], m) for pair in pairs
            ]
            current_generation = np.reshape(current_generation, (2 * (k_top - 1), n))

            # Mutation
            current_generation = [
                mutation(solution, m, m_factor) for solution in current_generation
            ]

            aux_gen_repr = (
                Generation(current_generation).move_to(UP * 0.8 + LEFT * 4.3).scale(0.2)
            )
            self.play(ReplacementTransform(survivals_repr, aux_gen_repr))

            current_generation_repr = aux_gen_repr

            # calculate fitness for this generation
            diversity_arr = [calculate_diversity(s, D) for s in current_generation]

            # for plotting
            best_solution_gen_history.append(sorted_gen_div[0][1])

            if current_best_solution_d < sorted_gen_div[0][1]:
                current_best_solution_d = sorted_gen_div[0][1]
                current_best_solution = sorted_gen_div[0][0]
                counter = 0

            if last_best_solution == current_best_solution_d:
                counter += 1
            else:
                last_best_solution = current_best_solution_d

            print(
                "> Best solution so far has diversity {}\n".format(last_best_solution)
            )

            self.play(
                ChangeDecimalToValue(
                    gen_counter, i, kwargs={"font": "SF Mono", "color": YELLOW}
                ),
                ChangeDecimalToValue(best_div, last_best_solution),
            )

            # for plotting
            best_solution_history.append(last_best_solution)
            patience_history.append(counter)

            print(
                "> Patience counter: {}. {} more to finish if equal.".format(
                    counter, patience - counter
                )
            )
            if counter == patience:
                print("\nPatience counter dropped to 0.\n")
                print(
                    "\nBest solution found had diversity {}".format(
                        current_best_solution_d
                    )
                )
                print("\nBest solution was {}".format(current_best_solution))
                return (current_best_solution, current_best_solution_d), (
                    best_solution_gen_history,
                    best_solution_history,
                    patience_history,
                )

        print("\nRan out of iterations.\n")
        print("\nBest solution found had diversity {}".format(current_best_solution_d))
        print("\nBest solution was {}".format(current_best_solution))
