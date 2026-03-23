import logging

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import jax

jax.config.update("jax_enable_x64", True)

from flax import struct
import jax.numpy as jnp
import jaxopt

import sys

if "jax" in sys.modules and sys.modules["jax"] is not jax:
    raise RuntimeError(
        "Do not import JAX directly. Always import through _jax_bootstrap."
    )

__all__ = ["jax", "jnp", "jaxopt", "struct"]
