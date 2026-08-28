"""Serialization utilities for PyTrees with optional metadata."""

import base64
import dataclasses
import json
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def save(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with optional metadata to a file.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    with open(path, "wb") as f:
        if not metadata:
            metadata = {}
        f.write(json.dumps(metadata, indent=None).encode("utf-8"))
        f.write(b"\n")
        eqx.tree_serialise_leaves(f, model)


def save_with_treedef(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with treedef and optional metadata to a file.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    # 1. Get the structure
    treedef = jax.tree.structure(model)

    # 2. Serialize the treedef using pickle, then base64 encode it to store in JSON
    # PyTreeDefs are C++ objects and cannot be purely JSON serialized.
    treedef_bytes = pickle.dumps(treedef)
    treedef_b64 = base64.b64encode(treedef_bytes).decode("ascii")

    if not metadata:
        metadata = {}
    metadata["treedef_b64"] = treedef_b64

    # 3. Write the file
    with open(path, "wb") as f:
        # Write JSON header followed by a newline
        header = json.dumps(metadata).encode("utf-8")
        f.write(header)
        f.write(b"\n")

        # Write the leaves (weights/arrays) using Equinox
        eqx.tree_serialise_leaves(f, model)


def load_metadata(path: Path | str) -> dict:
    """Load metadata from a file.

    Args:
        path (Path): The path to the file from which to load the metadata.

    Returns:
        dict: The loaded metadata.
    """
    with open(path, "rb") as f:
        metadata = json.loads(f.readline().decode("utf-8"))
    return metadata


def load(path: Path | str, template: PyTree) -> PyTree:
    """Load a PyTree from a file.

    Args:
        path (Path): The path to the file from which to load the PyTree.
        template (PyTree): A template PyTree with the same structure as the one being loaded.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        f.readline()  # Discard metadata line
        model = eqx.tree_deserialise_leaves(f, template)
    return model


def load_from_treedef(path: Path | str) -> PyTree:
    """Load a PyTree from a file from the stored treedef.

    Args:
        path (Path | str): The path to the file from which to load the PyTree.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        # 1. Read the header
        header_line = f.readline().strip()
        metadata = json.loads(header_line)

        # 2. Decode the treedef
        treedef_b64 = metadata["treedef_b64"]
        treedef_bytes = base64.b64decode(treedef_b64)
        try:
            treedef = pickle.loads(treedef_bytes)
        except ModuleNotFoundError as error:
            # Equinox 0.13.8 moved its internal flattening sentinel to a new
            # module. Treedefs written by that release cannot be unpickled by
            # 0.13.2, which is the version used by this project.
            if error.name != "equinox._module._flatten":
                raise
            return _load_semantic_ldba_leaves(f)

        # 3. Load the leaves
        # Since we don't have a template, we cannot use eqx.tree_deserialise_leaves.
        # Instead, we know how many leaves the treedef expects, and we know
        # eqx saves them as concatenated .npy files.
        leaves = []
        for _ in range(treedef.num_leaves):
            # jnp.load can read directly from the open file handle
            leaves.append(jnp.load(f))

        # 4. Reconstruct the model
        try:
            model = jax.tree.unflatten(treedef, leaves)
        except TypeError as error:
            # Equinox's private `_Missing` sentinel is stored in the pickled
            # PyTreeDef. Pickle recreates it as a different object, so recent
            # Equinox versions fail while unflattening a tree they just wrote.
            if "'_Missing' object is not subscriptable" not in str(error):
                raise
            f.seek(len(header_line) + 1)
            return _load_semantic_ldba_leaves(f)
    return _reinstantiate(model)


def _load_semantic_ldba_leaves(file) -> PyTree:
    """Load a legacy or state-acceptance JaxSemanticLDBA payload.

    This compatibility path avoids Equinox's private, version-dependent
    PyTreeDef metadata while preserving the array payload.
    """
    from jaxltl.semltl.utils.jax_semantic_ldba import JaxSemanticLDBA

    leaves = [jnp.load(file) for _ in range(8)]
    state_shape = leaves[4].shape
    accepting_states = jnp.zeros(state_shape, dtype=jnp.bool)
    accepting_sink_states = jnp.zeros(state_shape, dtype=jnp.bool)
    rejecting_sink_states = leaves[4].astype(jnp.bool)
    try:
        accepting_states = jnp.load(file)
        accepting_sink_states = jnp.load(file)
        rejecting_sink_states = jnp.load(file)
    except (EOFError, ValueError):
        # Old eight-array files contain no finite-word state metadata. Preserve
        # their known rejecting sinks and leave state/true-sink acceptance off.
        pass
    return JaxSemanticLDBA(
        num_states=leaves[0],
        initial_state=leaves[1],
        transitions=leaves[2],
        accepting=leaves[3],
        sink_states=leaves[4],
        finite=leaves[5],
        epsilon_transitions=leaves[6],
        embeddings=leaves[7],
        accepting_states=accepting_states,
        accepting_sink_states=accepting_sink_states,
        rejecting_sink_states=rejecting_sink_states,
    )


def _reinstantiate(tree: PyTree) -> PyTree:
    """Recursively reinstantiate Equinox modules in a PyTree to ensure the pickled
    class matches the current class definition.

    Args:
        tree (PyTree): The input PyTree.

    Returns:
        PyTree: The re-instantiated PyTree."""
    if isinstance(tree, eqx.Module):
        cls = type(tree)
        init_kwargs = {}
        for field in dataclasses.fields(tree):
            if field.init:
                value = getattr(tree, field.name)
                init_kwargs[field.name] = _reinstantiate(value)
        return cls(**init_kwargs)
    elif _is_named_tuple_instance(tree):
        cls = type(tree)
        init_kwargs = {}
        for field in tree._fields:
            value = getattr(tree, field)
            init_kwargs[field] = _reinstantiate(value)
        return cls(**init_kwargs)
    elif isinstance(tree, list | tuple):
        return type(tree)(_reinstantiate(x) for x in tree)
    elif isinstance(tree, dict):
        return {k: _reinstantiate(v) for k, v in tree.items()}
    else:
        return tree


def _is_named_tuple_instance(x):
    """Check if x is an instance of a namedtuple."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] is not tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) is str for n in f)
