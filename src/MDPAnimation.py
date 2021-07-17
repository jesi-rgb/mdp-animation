from manim import *
import numpy as np


np.random.seed(1)
# Distance Matrix
n = 5
D = np.random.randint(100, size=(n, n), dtype=int)


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
        self.play(FadeIn(Text("Maximum Diversity Problem", font="SF Mono")))
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

        squares = VGroup(
            *[
                Square(color=random_bright_color(), fill_opacity=1)
                .scale(0.5)
                .move_to((RIGHT + 0.1) * i, coor_mask=[1, 0, 0])
                for i in range(n)
            ]
        ).move_to(ORIGIN)

        self.play(FadeIn(squares))

        surround_sq_1 = SurroundingRectangle(squares[1])
        surround_sq_2 = SurroundingRectangle(squares[3])
        idx = coords_to_index(1, 3, n)
        surround_mx_1 = SurroundingRectangle(distance_matrix[idx])

        self.play(Create(surround_sq_1), Create(surround_sq_2), Create(surround_mx_1))
