import logging
import signal
import time

import hydra

import jaxltl
from jaxltl.deep_ltl.eval.utils import compute_cached_ldba_and_sequences
from jaxltl.ltl.automata import ltl2ldba, ltl2ldba_semml
from jaxltl.ltl.automata.ldba import LDBA
from jaxltl.ltl.logic.assignment import Assignment
from src.jaxltl.deep_ltl.reach_avoid import path_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timeout_handler(signum, frame):
    del signum, frame  # Unused.
    raise TimeoutError()


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval")
def main(_):
    env, _ = jaxltl.make("LetterWorld")
    formulas = [
        "GF(a) & G(a => F(g | j)) & G(g => F(l | b)) & G(j => F(k | f)) & G(l => F(d | h)) & G(b => F(c | e)) & G(k => F(i))"
    ]
    props = env.propositions
    assignments = tuple(env.assignments)

    signal.signal(signal.SIGALRM, timeout_handler)

    for i, formula in enumerate(formulas):
        timeout = 600  # seconds
        signal.alarm(timeout)

        try:
            start = time.time()
            ldba, seqs = compute_cached_ldba_and_sequences(formula, props, assignments)
            num_transitions = sum(
                [len(t_list) for t_list in ldba.state_to_transitions.values()]
            )
            logger.info(
                f"Formula {i}: LDBA with {len(ldba.states)} states and {num_transitions} transitions."
            )
            res = path_search.compute_sequences(ldba, num_loops=2)
            print(len(res[0]))
            end = time.time()
            inf_time = end - start

        except TimeoutError:
            inf_time = timeout
            logger.info(f"Formula {i}: Timeout reached ({timeout}s)")
        else:
            logger.info(f"Formula {i}: Inference time: {inf_time:.2f}s")
        finally:
            signal.alarm(0)


def build_ldba(
    formula: str,
    propositions: tuple[str, ...],
    assignments: tuple[Assignment, ...],
    semml: bool = False,
) -> LDBA:
    """Builds and preprocesses an LDBA from an LTL formula."""
    if semml:
        ldba = ltl2ldba_semml(formula, list(propositions), list(assignments))
    else:
        ldba = ltl2ldba(formula, propositions)
    ldba.prune(list(assignments))
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


if __name__ == "__main__":
    main()
