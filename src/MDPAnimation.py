from manimlib import *
import numpy as np

np.random.seed(1)
# Distance Matrix
n = 5
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


class IntroButterflies(Scene):
    def construct(self):

        intro_btf = Butterfly().get()

        self.play(FadeIn(intro_btf, RIGHT), run_time=1)
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
