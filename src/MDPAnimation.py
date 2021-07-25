from manimlib import *
import numpy as np
import networkx as nx

np.random.seed(1)
# Distance Matrix
n = 5
D = np.random.randint(100, size=(n, n), dtype=int)
np.fill_diagonal(D, 0)
D = np.triu(D)

TITLE = Text("Maximum Diversity Problem", font="SF Mono")
MDP_FORMULA = (
	Tex(r"MD(x) = \sum^{n-1}_{i=0} \sum^{n}_{j=i+1} D_{ij}x_ix_j")
)


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


class IntroButterflies(Scene):
    def construct(self):
        