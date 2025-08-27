"""Consolidated tests for basic scene generators that just verify basic functionality."""

from tests.map.scenes.utils import render_scene


class TestBasicSceneGenerators:
    """Tests for scene generators that just need basic functionality verification."""

    def test_nop_generator(self):
        """Test no-operation scene generator creates empty grid."""
        from metta.map.scenes.nop import Nop

        scene = render_scene(Nop.factory(), (3, 3))
        assert (scene.grid == "empty").sum() == 9

    def test_random_generator_objects(self):
        """Test random generator places correct number of objects."""
        from metta.map.scenes.random import Random

        scene = render_scene(
            Random.factory(Random.Params(objects={"altar": 3, "temple": 2})),
            (3, 3),
        )
        assert (scene.grid == "altar").sum() == 3
        assert (scene.grid == "temple").sum() == 2

    def test_random_generator_agents(self):
        """Test random generator places agents correctly."""
        from metta.map.scenes.random import Random

        scene = render_scene(Random.factory(Random.Params(agents=2)), (3, 3))
        assert (scene.grid == "agent.agent").sum() == 2

    def test_wfc_generator_basic(self):
        """Test Wave Function Collapse generator produces output."""
        from metta.map.scenes.wfc import WFC

        scene = render_scene(
            WFC.factory(
                WFC.Params(
                    pattern="""
                        .#...
                        ###..
                        ###..
                    """
                )
            ),
            (20, 20),
        )
        assert (scene.grid == "wall").sum() > 0
        assert (scene.grid == "empty").sum() > 0

    def test_convchain_generator_basic(self):
        """Test ConvChain generator produces output."""
        from metta.map.scenes.convchain import ConvChain

        scene = render_scene(
            ConvChain.factory(
                ConvChain.Params(
                    pattern="""
                        ##..#
                        #....
                        #####
                    """,
                    pattern_size=3,
                    iterations=10,
                    temperature=1,
                )
            ),
            (20, 20),
        )
        assert (scene.grid == "wall").sum() > 0
        assert (scene.grid == "empty").sum() > 0
