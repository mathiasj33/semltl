import random

from jaxltl.ltl2action.curriculum.curriculum import Sampler


class ReactiveSampler(Sampler[str]):
    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        self.depth = depth
        self.reach = reach
        self.propositions = propositions

    def sample(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        start = random.choice(self.propositions)
        formula = f"GF({start})"
        available = [p for p in self.propositions if p != start]
        remaining = [start]
        for _ in range(depth):
            new_remaining = []
            for i, prop in enumerate(remaining):
                if not available:
                    new_remaining.extend(remaining[i:])
                    break
                nr = random.randint(self.reach[0], self.reach[1])
                reach_props = random.sample(available, min(nr, len(available)))
                available = [p for p in available if p not in reach_props]
                new_remaining.extend(reach_props)
                formula += f" & G({prop} => F({' | '.join(reach_props)}))"
            remaining = new_remaining
            if not available:
                break
        # for prop in remaining:
        #     formula += f" & G({prop} => F({start}))"
        return formula


class FiniteReactiveSampler(Sampler[str]):
    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        self.depth = depth
        self.reach = reach
        self.propositions = propositions

    def sample(self) -> str:
        goal = random.choice(self.propositions)
        formula = f"{goal}"
        propositions = [p for p in self.propositions if p != goal]
        depth = random.randint(self.depth[0], self.depth[1])
        implications = []
        for _ in range(depth):
            nr = random.randint(self.reach[0], self.reach[1])
            reach_props = random.sample(propositions, min(nr, len(propositions)))
            implication_prop = random.choice(
                [p for p in propositions if p not in reach_props]
            )
            implications.append(f"({implication_prop} => F({' | '.join(reach_props)}))")
        return f"({' & '.join(implications)}) U ({formula})"


class GlobalSafetySampler(Sampler[str]):
    """Samples global safety formulas."""

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        self.depth = depth
        self.reach = reach
        self.propositions = propositions

    def sample(self, avoid: str | None) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        if not avoid:
            avoid = random.choice(self.propositions)
        propositions = [p for p in self.propositions if p != avoid]
        props = []
        last_props = set()
        for _ in range(depth):
            nr = random.randint(self.reach[0], self.reach[1])
            available_props = [p for p in propositions if p not in last_props]
            reach_props = random.sample(available_props, min(nr, len(available_props)))
            props.append(reach_props)
            last_props = set(reach_props)
        formula = "true"
        for reach_props in reversed(props):
            formula = f"(!{avoid} U (({' | '.join(reach_props)}) & {formula}))"
        return formula


class ConjunctiveReachSampler(Sampler[str]):
    def __init__(
        self,
        depth: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        self.depth = depth
        self.propositions = propositions

    def sample(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        props = random.sample(self.propositions, depth)
        formula = f"!{props[1]} U {props[0]}"
        for i in range(1, depth - 1):
            formula = f"!{props[i + 1]} U ({props[i]} & F ({formula}))"
        formula = f"F ({props[-1]} & F ({formula}))"
        return formula


class ComplexPatrolSampler(Sampler[str]):
    def __init__(
        self,
        depth: int | tuple[int, int],
        num_avoid: int,
        num_disjuncts: int,
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        self.depth = depth
        self.num_avoid = num_avoid
        self.num_disjuncts = num_disjuncts
        self.propositions = propositions

    def sample(self) -> str:
        disjuncts = [self._sample_disjunct() for _ in range(self.num_disjuncts)]
        formula = " | ".join(disjuncts)
        return formula

    def _sample_disjunct(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        avoid = random.sample(self.propositions, self.num_avoid)
        available = [p for p in self.propositions if p not in avoid]
        props = random.sample(available, depth)
        formula = f"F {props[0]}"
        for i in range(1, depth):
            formula = f"F ({props[i]} & ({formula}))"
        formula = f"G {formula} & {' & '.join([f'G (!{a})' for a in avoid])}"
        return formula


class FGSampler(Sampler[str]):
    def __init__(
        self,
        depth: int,
        propositions: list[str],
    ):
        self.depth = depth
        self.propositions = propositions

    def sample(self) -> str:
        props = random.sample(self.propositions, self.depth)
        formula = f"FG({props[0]})"
        for p in props[1:]:
            formula += f" & (F {p})"
        return formula


if __name__ == "__main__":
    # props = ("blue", "orange", "green", "red", "purple", "brown", "pink", "gray")
    props = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l")
    sampler = FGSampler(
        depth=5,
        propositions=list(props),
    )
    for _ in range(10):
        print(f' - "{sampler.sample()}"')
