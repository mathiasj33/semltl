import hydra
from omegaconf import DictConfig

from jaxltl import DATA_DIR
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.normalize_reward_wrapper import NormalizeRewardWrapper
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxltl.semltl.wrappers.semantic_dwa_wrapper import SemanticDWAWrapper
from jaxltl.semltl.wrappers.semantic_ldba_wrapper import SemanticLDBAWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper:
    overwrite_finite = cfg.get("eval", {}).get("finite", False)
    env = SemanticLDBAWrapper(env, overwrite_finite=overwrite_finite)
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
        env = NormalizeRewardWrapper(env, gamma=cfg.rl_alg.gamma)
    return env


def wrap_dwa_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper:
    """Wrap an environment with finite-trace FishSemML DWA semantics."""
    buchi_rewards = bool(
        not training and cfg.get("eval", {}).get("buchi_rewards", False)
    )
    env = SemanticDWAWrapper(env, buchi_rewards=buchi_rewards)
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
        env = NormalizeRewardWrapper(env, gamma=cfg.rl_alg.gamma)
    return env
