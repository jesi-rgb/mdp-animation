from manimlib import *
import numpy as np

from mdp import (
    calculate_diversity,
    create_random_solution,
    two_point_crossover,
    mutation,
)
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(1)

# Distance Matrix
n = 10
m = 5
D = np.random.randint(100, size=(n, n), dtype=int)
np.fill_diagonal(D, 0)
D = np.triu(D)

# TITLE = Text("Maximum Diversity Problem", font="SF Mono")
# MDP_FORMULA = Tex(r"MD(x) = \sum^{n-1}_{i=0} \sum^{n}_{j=i+1} D_{ij}x_ix_j")


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
            self.square.set_fill(BLACK, opacity=1)
            self.number.set_color(WHITE)
        else:
            self.square.set_fill(WHITE, opacity=1)
            self.number.set_color(BLACK)

        super().__init__(self.square, self.number)
        self.arrange(ORIGIN, buff=0)


class Solution(VGroup):
    def __init__(self, M):
        self.squares = VGroup()
        for i in range(len(M)):
            self.squares.add(SolutionSquare(M[i]))

        super().__init__(self.squares)
        self.squares.arrange_in_grid(n_rows=1, buff=0.01)


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
        solution = [1, 0, 0, 1, 0, 1]
        self.add(Solution(solution))


class GeneticAlgorithm(Scene):
    def construct(self):

        initial_population = 10
        k_top = 15
        m_factor = 0.002
        n_iterations = 500
        patience = 20

        current_generation = [
            create_random_solution(n, m) for _ in range(initial_population)
        ]

        diversity_arr = [calculate_diversity(s, D) for s in current_generation]

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
            sorted_gen_div = sorted(gen_div, key=lambda x: x[1], reverse=True)

            # Select k_top best parents from this generation
            best_solutions = [s[0] for s in sorted_gen_div]
            survivals = best_solutions[:k_top]

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
