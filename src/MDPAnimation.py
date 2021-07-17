from manim import *
import numpy as np


np.random.seed(1)
# Distance Matrix
n = 5
D = np.random.randint(100, size=(n, n), dtype=int)
np.fill_diagonal(D, 0)


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

        self.wait()

        # DEMONSTRATION OF THE DISTANCE BETWEEN THE SQUARES

        surround_sq_1 = SurroundingRectangle(squares[1])
        surround_sq_2 = SurroundingRectangle(squares[3])
        idx = coords_to_index(1, 3, n)
        surround_mx_1 = SurroundingRectangle(distance_matrix[idx])

        self.play(Create(surround_sq_1), Create(surround_sq_2), Create(surround_mx_1))

        self.wait()

        surround_sq_3 = SurroundingRectangle(squares[2])
        surround_sq_4 = SurroundingRectangle(squares[4])
        idx = coords_to_index(2, 4, n)
        surround_mx_2 = SurroundingRectangle(distance_matrix[idx])

        self.play(
            ReplacementTransform(surround_sq_1, surround_sq_3),
            ReplacementTransform(surround_sq_2, surround_sq_4),
            ReplacementTransform(surround_mx_1, surround_mx_2),
        )
        self.wait()

        surround_sq_5 = SurroundingRectangle(squares[0])
        surround_sq_6 = SurroundingRectangle(squares[3])
        idx = coords_to_index(0, 3, n)
        surround_mx_3 = SurroundingRectangle(distance_matrix[idx])

        self.play(
            ReplacementTransform(surround_sq_3, surround_sq_5),
            ReplacementTransform(surround_sq_4, surround_sq_6),
            ReplacementTransform(surround_mx_2, surround_mx_3),
        )
        self.wait()

        surround_sq_7 = SurroundingRectangle(squares[1])
        surround_sq_8 = SurroundingRectangle(squares[2])
        idx = coords_to_index(1, 2, n)
        surround_mx_4 = SurroundingRectangle(distance_matrix[idx])

        self.play(
            ReplacementTransform(surround_sq_5, surround_sq_7),
            ReplacementTransform(surround_sq_6, surround_sq_8),
            ReplacementTransform(surround_mx_3, surround_mx_4),
        )
        self.wait()

        surround_sq_9 = SurroundingRectangle(squares[4])
        surround_sq_10 = SurroundingRectangle(squares[4])
        idx = coords_to_index(4, 4, n)
        surround_mx_5 = SurroundingRectangle(distance_matrix[idx])

        self.play(
            ReplacementTransform(surround_sq_7, surround_sq_9),
            ReplacementTransform(surround_sq_8, surround_sq_10),
            ReplacementTransform(surround_mx_4, surround_mx_5),
        )
        self.wait()
