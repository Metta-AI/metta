import os
import signal  # Aggressively exit on ctrl+c
import json # For loading historical elo or glicko scores from local disk
import hydra
from omegaconf import OmegaConf
from rich import traceback
from util.eval_analyzer import Analysis
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg, policy_store)
        stats = evaluator.evaluate()
        evaluator.close()

        p_analysis = Analysis(stats, eval_method='1v1', stat_category='all')

        # Get raw results
        p_results = p_analysis.get_results()
        print(p_results) # Printing raw results for demonstration purposes

        # Print formatted results
        print(p_analysis.get_display_results())

        #--------Elo Analysis--------#
        # Load the historical elo scores from disk
        with open("evals/elo_scores.json", "r") as file:
            historical_elo_scores = json.load(file)

        elo_analysis = Analysis(stats, eval_method='elo_1v1', stat_category='altar', initial_elo_scores=historical_elo_scores)
        # Get raw results
        elo_results = elo_analysis.get_results()

        # Print formatted results
        print(elo_analysis.get_display_results())

        updated_elo_scores = elo_analysis.get_updated_historicals()
        print(updated_elo_scores)

        # Overwrite the JSON file with updated elo scores
        with open("evals/elo_scores.json", "w") as file:
            json.dump(updated_elo_scores, file, indent=4)
        #------End Elo Analysis--------#
        
        #--------Glicko Analysis--------#
        with open("evals/glicko_scores.json", "r") as file:
            historical_glicko_scores = json.load(file)

        glicko_analysis = Analysis(
            data=stats,
            eval_method='glicko2_1v1',
            stat_category='action.use.altar',
            initial_glicko2_scores=historical_glicko_scores
        )

        # print(glicko_analysis.get_results())
        print(glicko_analysis.get_display_results())

        updated_glicko_scores = glicko_analysis.get_updated_historicals()
        with open("evals/glicko_scores.json", "w") as file:
            json.dump(updated_glicko_scores, file, indent=4)

        from pprint import pprint

        print("Old glicko scores:")
        if historical_glicko_scores:
            pprint(historical_glicko_scores)

        print("New glicko scores:")
        pprint(updated_glicko_scores)

        verbose_results = glicko_analysis.get_verbose_results()
        print("Verbose results:")
        pprint(verbose_results)




if __name__ == "__main__":
    main()
