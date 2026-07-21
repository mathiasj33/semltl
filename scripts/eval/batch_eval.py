"""Evaluate trained models on a specified set of LTL formulas.

Evaluates all models on all formulas. Saves results to a CSV
file and prints to stdout. Use the batch_size config options
to trade off speed and memory usage during evaluation.

Relies on precomputing algorithm-specific structures (e.g. LTL
formula closure or automata paths) to enable fast evaluation
in Jax. Not suitable for large formulas, or to run precise
timing experiments.
"""

import csv
import logging
import os
import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from omegaconf import DictConfig

import jaxltl
from jaxltl import eqx_utils
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxltl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxltl.eval.utils import (
    load_batched_models,
    make_eval_fn,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="batch_eval")
def main(cfg: DictConfig):
    # build environment
    env_params = cfg.get("env_params", {})
    env, env_params = jaxltl.make(cfg.env.name, **env_params)
    if cfg.env.use_precomputed_resets:
        resets_path = (
            f"{jaxltl.DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        )
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    formulas: PyTree = hydra.utils.call(cfg.alg.preprocess_formulas, cfg.formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, num_models = load_batched_models(cfg, env, env_params, key=model_key)
    agents = hydra.utils.instantiate(cfg.alg.agent, models)

    # log num parameters
    params, _ = eqx.partition(models, eqx.is_array)
    params = jax.tree.map(lambda x: x[0], params)  # take first model
    num_params = eqx_utils.count_parameters(params)
    logger.info(
        f"Loaded models with number of parameters per model: {num_params / 1e3:.2f}K"
    )

    # set up evaluator
    eval_fn = make_eval_fn(
        cfg, num_models, num_formulas=len(cfg.formulas), return_trajs=False
    )

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    returns, disc_returns, lengths, _, info = eval_fn(
        agents,
        env,
        env_params,
        formulas,
        eval_key,
    )  # shape: (num_seeds, num_formulas, num_episodes)
    jax.block_until_ready(returns)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    log_and_save_results(cfg, returns, lengths)

    if cfg.alg.name == "semltl":
        num_states = info["num_visited_ldba_states"]
        successes = returns > 0
        num_states = num_states * successes
        avg_states = num_states.sum(axis=-1) / successes.sum(axis=-1)
        per_seed_avg_states = jnp.nanmean(avg_states, axis=-1)
        logger.info("========================================")
        logger.info("SemLTL-specific stats:")
        logger.info(
            f"Average number of visited LDBA states: {float(jnp.mean(per_seed_avg_states)):.2f}+-{float(jnp.std(per_seed_avg_states)):.2f}"
        )
        per_formula_means = jnp.mean(avg_states, axis=0)
        per_formula_std = jnp.std(avg_states, axis=0)
        for i, formula in enumerate(cfg.formulas):
            logger.info(
                f"Formula: {formula} | Avg visited LDBA states: {float(per_formula_means[i]):.2f}+-{float(per_formula_std[i]):.2f}"
            )


def log_and_save_results(cfg: DictConfig, returns: jax.Array, lengths: jax.Array):
    """Logs aggregated results per formula and saves per-seed results to a CSV file."""
    csv_path = f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/eval_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    num_seeds = int(returns.shape[0])
    seeds = list(range(num_seeds))

    fieldnames = [
        "seed",
        "deterministic",
        "formula",
        "return",
        "length",
    ]

    rows = []
    for i, formula in enumerate(cfg.formulas):
        # Compute per-seed stats
        returns_i = returns[:, i]  # (num_seeds, num_episodes)
        lengths_i = lengths[:, i]  # (num_seeds, num_episodes)

        means = jnp.mean(returns_i, axis=1)  # (num_seeds,)

        success_mask = returns_i > 0  # (num_seeds, num_episodes)
        success_counts = jnp.sum(success_mask, axis=1)  # (num_seeds,)
        sum_lengths = jnp.sum(lengths_i * success_mask, axis=1)
        avg_lengths = jnp.where(
            success_counts > 0, sum_lengths / success_counts, jnp.nan
        )

        # Stdout logging (aggregate across seeds)
        logger.info("========================================")
        logger.info(f"Formula: {formula}")
        logger.info(f"SR/AV: {float(jnp.mean(means)):.3f}+-{float(jnp.std(means)):.3f}")
        logger.info(
            f"Length: {float(jnp.mean(avg_lengths)):.3f}+-{float(jnp.std(avg_lengths)):.3f}"
        )

        # CSV rows (per-seed)
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "formula": formula,
                    "return": float(means[seed]),
                    "length": float(avg_lengths[seed]),
                }
            )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote results to {csv_path}")

    per_seed_means = jnp.mean(returns, axis=(1, 2))  # (num_seeds,)
    logger.info("========================================")
    logger.info(
        f"Overall SR/AV: {float(jnp.mean(per_seed_means)):.3f}+-{float(jnp.std(per_seed_means)):.3f}"
    )


if __name__ == "__main__":
    main()
