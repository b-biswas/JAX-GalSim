import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_galsim


def test_image_jax_view_raises():
    im = jax_galsim.ImageD(np.arange(20).reshape(4, 5))
    with pytest.raises(NotImplementedError):
        im.view()


def test_image_dtype_tracks_storage_with_x64_disabled():
    original_x64 = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.warns(UserWarning, match="truncated to dtype float32"):
            image = jax_galsim.Image(2, 2, dtype=jnp.float64)

        assert image.dtype == image.array.dtype.type

        leaves, treedef = jax.tree_util.tree_flatten(image)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.dtype == rebuilt.array.dtype.type
    finally:
        jax.config.update("jax_enable_x64", original_x64)
