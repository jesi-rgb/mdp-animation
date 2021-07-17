from manim import *
import numpy as np
import networkx as nx

np.random.seed(1)
# Distance Matrix
n = 5
D = np.random.randint(100, size=(n, n), dtype=int)
np.fill_diagonal(D, 0)
D = np.triu(D)


class MMatrix:
    def __init__(self, D) -> None:
        numbers_to_text = [
            Text(f"{num}", font="SF Mono", color=YELLOW) for num in D.ravel()
        ]
        dist_matrix = VGroup(*numbers_to_text)
        dist_matrix.arrange_in_grid(n_rows=n, buff=0.4)
        self.mmatrix = dist_matrix

    def get_mmatrix(self) -> VGroup:
        return self.mmatrix


######################### ANIMATIONS


class Intro(Scene):
    def construct(self):
        title = Text("Maximum Diversity Problem", font="SF Mono")
        formula = (
            MathTex(r"MD(x) = \sum^{n-1}_{i=0} \sum^{n}_{j=i+1} D_{ij}x_ix_j")
            .move_to(2 * DOWN)
            .scale(0.6)
        )
        group = VGroup(*[title, formula]).move_to(ORIGIN)
        self.play(LaggedStartMap(FadeIn, group))
        self.wait()


def coords_to_index(i, j, n):
    return n * i + j


class MDPAnimation(Scene):
    def construct(self):
        title_matrix_distance = Text("Distance Matrix", font="Avenir")
        self.play(FadeIn(title_matrix_distance))
        self.play(title_matrix_distance.animate.shift(UP * 2))

        distance_matrix = MMatrix(D).get_mmatrix().next_to(title_matrix_distance, DOWN)
        dst_m_sm = distance_matrix.copy().scale(0.3).move_to([-6, 3, 0])

        self.play(LaggedStartMap(FadeIn, distance_matrix))

        self.play(
            ReplacementTransform(distance_matrix, dst_m_sm),
            FadeOut(title_matrix_distance),
        )
        self.remove(title_matrix_distance)

        G = nx.complete_graph(n)

        graph_og = Graph(
            vertices=list(G.nodes()),
            edges=list(G.edges()),
            layout="circular",
            labels=True,
        )

        graph = graph_og.copy()

        self.play(Create(graph))

        self.wait()

        # DEMONSTRATION OF THE DISTANCE BETWEEN THE DOTS

        idx = coords_to_index(1, 3, n)
        surround_mx_1 = SurroundingRectangle(distance_matrix[idx])

        self.play(
            graph[1].animate.set_color(YELLOW),
            graph[3].animate.set_color(YELLOW),
            graph.edges[(1, 3)].animate.set_color(YELLOW),
            Create(surround_mx_1),
        )

        self.wait()

        n_vertex = list(range(n))
        combs = set(([(i, j) for i in n_vertex for j in n_vertex if i < j]))

        surr_sq = [surround_mx_1]

        # TODO: try to speed up the animation of this, to go much faster
        for i, j in combs:

            idx = coords_to_index(i, j, n)
            curr_surr_sq = SurroundingRectangle(distance_matrix[idx])

            self.play(ReplacementTransform(graph, graph_og))

            self.play(
                graph[i].animate.set_color(YELLOW),
                graph[j].animate.set_color(YELLOW),
                graph.edges[(i, j)].animate.set_color(YELLOW),
                ReplacementTransform(surr_sq[-1], curr_surr_sq),
                run_time=1,
            )
            surr_sq.append(curr_surr_sq)

        self.wait()
