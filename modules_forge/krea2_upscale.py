import math


SAFE_DIFFUSION_LONG_EDGE = 3584


def multiple_of(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def require_positive_int(value: int | None, name: str):
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0.")


def target_size(
    width: int,
    height: int,
    long_edge: int,
    explicit_width: int | None,
    explicit_height: int | None,
) -> tuple[int, int]:
    if (explicit_width is None) != (explicit_height is None):
        raise ValueError("--width and --height must be passed together.")
    require_positive_int(width, "source width")
    require_positive_int(height, "source height")
    require_positive_int(long_edge, "--long-edge")
    require_positive_int(explicit_width, "--width")
    require_positive_int(explicit_height, "--height")

    if explicit_width is not None and explicit_height is not None:
        return multiple_of(explicit_width, 64), multiple_of(explicit_height, 64)
    if width >= height:
        return multiple_of(long_edge, 64), multiple_of(height * long_edge / width, 64)
    return multiple_of(width * long_edge / height, 64), multiple_of(long_edge, 64)


def size_from_long_edge(width: int, height: int, long_edge: int) -> tuple[int, int]:
    require_positive_int(width, "width")
    require_positive_int(height, "height")
    require_positive_int(long_edge, "long edge")

    if width >= height:
        return multiple_of(long_edge, 64), multiple_of(height * long_edge / width, 64)
    return multiple_of(width * long_edge / height, 64), multiple_of(long_edge, 64)


def capped_diffusion_size(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    cap_long_edge: int,
) -> tuple[int, int]:
    if cap_long_edge < 0:
        raise ValueError("diffusion long edge cap must be >= 0.")
    if cap_long_edge == 0 or max(target_width, target_height) <= cap_long_edge:
        return target_width, target_height

    source_long_edge = max(source_width, source_height)
    if cap_long_edge < source_long_edge:
        raise ValueError(
            "diffusion long edge cap must be >= source long edge when it caps the diffusion pass."
        )
    return size_from_long_edge(target_width, target_height, cap_long_edge)


def require_safe_diffusion_size(width: int, height: int, allow_unsafe: bool = False):
    require_positive_int(width, "diffusion width")
    require_positive_int(height, "diffusion height")
    if allow_unsafe:
        return

    long_edge = max(width, height)
    if long_edge > SAFE_DIFFUSION_LONG_EDGE:
        raise ValueError(
            f"diffusion long edge {long_edge} exceeds safe limit {SAFE_DIFFUSION_LONG_EDGE}. "
            "Set a smaller diffusion cap or explicitly allow unsafe large diffusion."
        )


def auto_first_pass_long_edge(
    source_width: int, source_height: int, final_width: int, final_height: int
) -> int:
    require_positive_int(source_width, "source width")
    require_positive_int(source_height, "source height")
    require_positive_int(final_width, "final width")
    require_positive_int(final_height, "final height")

    source_long_edge = max(source_width, source_height)
    final_long_edge = max(final_width, final_height)
    if final_long_edge <= source_long_edge:
        raise ValueError("final long edge must be larger than source long edge.")

    first_pass_long_edge = multiple_of(
        math.sqrt(source_long_edge * final_long_edge), 64
    )
    first_pass_long_edge = max(first_pass_long_edge, multiple_of(source_long_edge, 64))
    first_pass_long_edge = min(first_pass_long_edge, final_long_edge - 64)
    if first_pass_long_edge < source_long_edge:
        raise ValueError(
            "final long edge is too close to source long edge for a two-stage upscale."
        )
    return first_pass_long_edge


def two_stage_sizes(
    source_width: int,
    source_height: int,
    final_width: int,
    final_height: int,
    first_pass_long_edge: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    require_positive_int(source_width, "source width")
    require_positive_int(source_height, "source height")
    require_positive_int(final_width, "final width")
    require_positive_int(final_height, "final height")
    if first_pass_long_edge < 0:
        raise ValueError("first pass long edge must be >= 0.")

    source_long_edge = max(source_width, source_height)
    final_long_edge = max(final_width, final_height)

    if final_long_edge <= source_long_edge:
        raise ValueError("final long edge must be larger than source long edge.")
    if first_pass_long_edge == 0:
        first_pass_long_edge = auto_first_pass_long_edge(
            source_width, source_height, final_width, final_height
        )
    if first_pass_long_edge < source_long_edge:
        raise ValueError("first pass long edge must be >= source long edge.")
    if first_pass_long_edge >= final_long_edge:
        raise ValueError("first pass long edge must be < final long edge.")

    return size_from_long_edge(final_width, final_height, first_pass_long_edge), (
        final_width,
        final_height,
    )
