from jaxltl.environments.conveyor_world.conveyor_world import ConveyorWorld
from jaxltl.environments.environment import Environment, EnvParams
from jaxltl.environments.letter_world.letter_world import LetterWorld
from jaxltl.environments.zone_env8.zone_env8 import ZoneEnv8

_name_to_env = {
    "ZoneEnv8": ZoneEnv8,
    "LetterWorld": LetterWorld,
    "ConveyorWorld": ConveyorWorld,
}


def make(name: str, **kwargs) -> tuple[Environment, EnvParams]:
    """Create an environment by name.

    Returns:
        A tuple of the environment instance and its default parameters."""
    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class(**kwargs)
    return env, env.default_params  # type: ignore
