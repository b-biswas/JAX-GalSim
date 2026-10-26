import jax
import jax.numpy as jnp

from jax_galsim.image import Image
from jax_galsim.interpolatedimage import InterpolatedImage


def describe_image(label, image):
    print(f"{label}: Image.dtype={image.dtype}, Image.array.dtype={image.array.dtype}")


def check_mode(enable_x64):
    jax.config.update("jax_enable_x64", enable_x64)
    print(f"\njax_enable_x64={enable_x64}")

    image = Image(2, 2, dtype=jnp.float64)
    describe_image("requested float64", image)

    leaves, treedef = jax.tree_util.tree_flatten(image)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    describe_image("after PyTree roundtrip", rebuilt)

    copied = image.copy()
    describe_image("after copy", copied)
    print()

    float32_image = Image(jnp.ones((4, 4), dtype=jnp.float32), scale=1.0)
    fourier_image = float32_image.calculate_fft()
    inverse_image = fourier_image.calculate_inverse_fft()
    describe_image("float32 FFT", fourier_image)
    describe_image("inverse FFT", inverse_image)
    print()

    profile = InterpolatedImage(float32_image, scale=1.0)
    implementation = profile._original
    print(
        "InterpolatedImage: source array.dtype="
        f"{float32_image.array.dtype}, profile image array.dtype="
        f"{implementation._image.array.dtype}, image flux dtype="
        f"{implementation._image_flux.dtype}, flux ratio dtype="
        f"{jnp.asarray(implementation._flux_ratio).dtype}"
    )


def main():
    original_x64 = jax.config.read("jax_enable_x64")
    try:
        check_mode(False)
        check_mode(True)
    finally:
        jax.config.update("jax_enable_x64", original_x64)


if __name__ == "__main__":
    main()
