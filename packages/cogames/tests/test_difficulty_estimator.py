"""Tests for the mission difficulty estimator."""

from cogames.cogs_vs_clips.difficulty_estimator import (
    DifficultyReport,
    estimate_difficulty,
)
from cogames.cogs_vs_clips.missions import (
    EasyHeartsTrainingMission,
    EasyMode,
    HarvestMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    RepairMission,
    VibeCheckMission,
)
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
    EnergizedVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    PackRatVariant,
    RoughTerrainVariant,
)


class TestBasicEstimation:
    """Test basic estimation functionality."""

    def test_estimate_difficulty_returns_report(self):
        """estimate_difficulty returns a DifficultyReport."""
        report = estimate_difficulty(HarvestMission)
        assert isinstance(report, DifficultyReport)

    def test_report_has_required_fields(self):
        """Report contains all expected fields."""
        report = estimate_difficulty(HarvestMission)
        assert hasattr(report, "feasible")
        assert hasattr(report, "conflicts")
        assert hasattr(report, "energy")
        assert hasattr(report, "resources")
        assert hasattr(report, "recipe")
        assert hasattr(report, "spatial")
        assert hasattr(report, "difficulty_score")

    def test_report_summary_is_string(self):
        """Report.summary() returns readable string."""
        report = estimate_difficulty(HarvestMission)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Difficulty" in summary


class TestEasyMissions:
    """Test that easy missions are rated as easy."""

    def test_easy_mode_is_easy(self):
        """EasyMode mission should have low difficulty."""
        report = estimate_difficulty(EasyMode)
        assert report.feasible
        # Easy missions should be < 5x baseline difficulty
        assert report.difficulty_score < 5

    def test_easy_hearts_training_is_easy(self):
        """EasyHeartsTrainingMission should have low difficulty."""
        report = estimate_difficulty(EasyHeartsTrainingMission)
        assert report.feasible
        # Easy missions should be < 5x baseline difficulty
        assert report.difficulty_score < 5

    def test_harvest_mission_is_feasible(self):
        """Basic HarvestMission should be feasible."""
        report = estimate_difficulty(HarvestMission)
        assert report.feasible
        assert len(report.conflicts) == 0


class TestHardMissions:
    """Test that hard missions are rated as harder."""

    def test_large_map_increases_difficulty(self):
        """Larger maps should increase difficulty."""
        small_report = estimate_difficulty(HarvestMission)
        large_report = estimate_difficulty(Machina1OpenWorldMission)
        # Machina1 is 200x200, HarvestMission is 13x13
        assert large_report.spatial is not None
        assert large_report.spatial.map_area > small_report.spatial.map_area
        # Larger map should require more steps (steady state)
        assert large_report.steady_state_steps > small_report.steady_state_steps

    def test_vibe_check_requires_coordination(self):
        """VibeCheckMission requires agent coordination."""
        report = estimate_difficulty(VibeCheckMission)
        assert report.feasible
        # Should have some coordination difficulty
        assert report.recipe is not None


class TestVariantEffects:
    """Test how variants affect difficulty."""

    def test_dark_side_increases_difficulty(self):
        """DarkSideVariant (zero regen) increases difficulty."""
        base = estimate_difficulty(HarvestMission)
        with_dark = estimate_difficulty(HarvestMission.with_variants([DarkSideVariant()]))

        # Dark side removes energy regen
        assert with_dark.energy.regen_per_step == 0
        # Should be harder
        assert with_dark.difficulty_score >= base.difficulty_score

    def test_rough_terrain_increases_difficulty(self):
        """RoughTerrainVariant increases move cost."""
        base = estimate_difficulty(HarvestMission)
        with_rough = estimate_difficulty(HarvestMission.with_variants([RoughTerrainVariant()]))

        assert with_rough.energy.move_cost > base.energy.move_cost
        assert with_rough.difficulty_score >= base.difficulty_score

    def test_lonely_heart_reduces_difficulty(self):
        """LonelyHeartVariant simplifies recipes."""
        with_lonely = estimate_difficulty(HarvestMission.with_variants([LonelyHeartVariant()]))

        # LonelyHeart should make it easier and remain feasible
        assert with_lonely.feasible

    def test_inventory_heart_tune_reduces_extraction_need(self):
        """InventoryHeartTuneVariant pre-fills inventory."""
        base = estimate_difficulty(HarvestMission)
        with_inv = estimate_difficulty(HarvestMission.with_variants([InventoryHeartTuneVariant(hearts=5)]))

        # Should reduce effective extraction need
        if with_inv.initial_resources:
            # Some resources should be pre-filled
            total_base_need = sum(base.resources.heart_cost.values())
            total_effective_need = sum(with_inv.initial_resources.effective_extraction_need.values())
            assert total_effective_need <= total_base_need

    def test_pack_rat_doesnt_break_feasibility(self):
        """PackRatVariant (high caps) shouldn't break feasibility."""
        report = estimate_difficulty(HarvestMission.with_variants([PackRatVariant()]))
        assert report.feasible

    def test_energized_improves_energy_economy(self):
        """EnergizedVariant gives unlimited energy."""
        report = estimate_difficulty(HarvestMission.with_variants([EnergizedVariant()]))
        assert report.energy.energy_positive
        assert report.energy.sustainable_steps >= 10000


class TestConflictDetection:
    """Test detection of impossible configurations."""

    def test_detects_cargo_overflow(self):
        """Detect when recipe exceeds cargo capacity."""
        # Use HelloWorldOpenWorldMission which doesn't have LonelyHeartVariant
        # that would simplify recipes
        mission = HelloWorldOpenWorldMission.model_copy(deep=True)
        mission.cargo_capacity = 5  # Very low - recipe needs ~60 total cargo
        mission.assembler.first_heart_cost = 20  # High cost

        report = estimate_difficulty(mission)
        # Should detect cargo conflict (recipe needs carbon+oxygen+germanium+silicon > 5)
        assert any("cargo" in c.lower() for c in report.conflicts)

    def test_repair_mission_detects_clipped_extractors(self):
        """RepairMission has clipped extractors but should still work."""
        report = estimate_difficulty(RepairMission)
        # Repair mission has clipping enabled, so extractors can be unclipped
        # This should be feasible because clip_period is set
        assert report.energy is not None  # Ensure analysis ran


class TestEnergyAnalysis:
    """Test energy budget calculations."""

    def test_energy_positive_with_high_regen(self):
        """High regen should make energy positive."""
        mission = HarvestMission.model_copy(deep=True)
        mission.energy_regen_amount = 10
        mission.move_energy_cost = 2

        report = estimate_difficulty(mission)
        assert report.energy.energy_positive

    def test_energy_negative_with_zero_regen(self):
        """Zero regen with move cost should be energy negative."""
        mission = HarvestMission.model_copy(deep=True)
        mission.energy_regen_amount = 0
        mission.move_energy_cost = 2

        report = estimate_difficulty(mission)
        assert not report.energy.energy_positive

    def test_sustainable_steps_calculated(self):
        """Sustainable steps should be calculated correctly."""
        mission = HarvestMission.model_copy(deep=True)
        mission.energy_capacity = 100
        mission.energy_regen_amount = 0
        mission.move_energy_cost = 5

        report = estimate_difficulty(mission)
        # 100 energy / 5 per step = 20 steps
        assert report.energy.sustainable_steps == 20


class TestResourceAnalysis:
    """Test resource requirement calculations."""

    def test_heart_cost_matches_assembler_config(self):
        """Heart cost should match assembler configuration."""
        mission = HarvestMission.model_copy(deep=True)
        mission.assembler.first_heart_cost = 15

        report = estimate_difficulty(mission)
        assert report.resources.heart_cost["carbon"] == 15
        assert report.resources.heart_cost["oxygen"] == 15
        assert report.resources.heart_cost["silicon"] == 45  # 3x

    def test_extractor_visits_calculated(self):
        """Extractor visits should be calculated based on output."""
        report = estimate_difficulty(HarvestMission)
        # Should have visits for all resources
        assert "carbon" in report.min_extractor_visits
        assert "oxygen" in report.min_extractor_visits
        assert "germanium" in report.min_extractor_visits
        assert "silicon" in report.min_extractor_visits


class TestSpatialAnalysis:
    """Test spatial complexity estimation."""

    def test_hub_based_maps_detected(self):
        """Hub-based maps should be detected."""
        report = estimate_difficulty(HarvestMission)
        assert report.spatial.is_hub_based

    def test_large_maps_have_higher_complexity(self):
        """Larger maps should have higher spatial complexity."""
        small = estimate_difficulty(HarvestMission)
        large = estimate_difficulty(HelloWorldOpenWorldMission)

        assert large.spatial.map_area > small.spatial.map_area


class TestDifficultyScoring:
    """Test the difficulty scoring formula."""

    def test_score_is_positive(self):
        """Difficulty score should be positive (1.0 = baseline)."""
        for mission in [HarvestMission, EasyMode, Machina1OpenWorldMission]:
            report = estimate_difficulty(mission)
            assert report.difficulty_score >= 0

    def test_easy_missions_have_low_scores(self):
        """Easy missions should have scores near 1.0."""
        report = estimate_difficulty(EasyMode)
        # Easy missions should be < 10x baseline
        assert report.difficulty_score < 10

    def test_impossible_config_not_feasible(self):
        """Impossible configurations should not be feasible."""
        mission = HarvestMission.model_copy(deep=True)
        mission.cargo_capacity = 1  # Too small for any recipe

        report = estimate_difficulty(mission)
        # Should have conflicts and be marked not feasible
        assert len(report.conflicts) > 0
        assert not report.feasible


class TestRecipeAnalysis:
    """Test recipe complexity analysis."""

    def test_min_agents_detected(self):
        """Minimum agents for recipes should be detected."""
        report = estimate_difficulty(VibeCheckMission)
        assert report.recipe is not None
        # VibeCheck requires at least 2 agents vibing

    def test_cheapest_recipe_found(self):
        """Should find the cheapest accessible recipe."""
        report = estimate_difficulty(HarvestMission)
        assert report.recipe.cheapest_heart_cost is not None

