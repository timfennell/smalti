"""
Physical unit system for SVG output and UI display.
Working pixels are an internal unit; all user-facing values are in cm or inches.
"""

PX_PER_INCH = 96.0  # SVG spec default
CM_PER_INCH = 2.54

UNIT_LABELS = {
    'cm': 'cm',
    'in': 'in',
    'mm': 'mm',
    'px': 'px',
}

# 'px' is a pixel unit: the canvas size is given directly in output pixels rather than
# a physical measurement.  It is defined against PX_PER_INCH so that, at dpi=96, one
# working pixel maps 1:1 to one output pixel (used for video / screen output).  Keep
# dpi at 96 when unit == 'px'.
UNITS_TO_INCHES = {
    'in': 1.0,
    'cm': 1.0 / CM_PER_INCH,
    'mm': 0.1 / CM_PER_INCH,
    'px': 1.0 / PX_PER_INCH,
}


def to_inches(value: float, unit: str) -> float:
    return value * UNITS_TO_INCHES[unit]


def from_inches(value: float, unit: str) -> float:
    return value / UNITS_TO_INCHES[unit]


def to_svg_px(value_in_unit: float, unit: str, dpi: float = 96.0) -> float:
    """Convert a physical measurement to SVG user units (px at given DPI)."""
    inches = to_inches(value_in_unit, unit)
    return inches * dpi


def working_px_to_svg(
    working_px: float,
    working_size_px: int,
    physical_size: float,
    unit: str,
    dpi: float = 96.0,
) -> float:
    """
    Scale a working-pixel coordinate to SVG px.
    working_size_px: longest edge of working image in pixels
    physical_size: longest edge in physical units
    """
    svg_total = to_svg_px(physical_size, unit, dpi)
    scale = svg_total / working_size_px
    return working_px * scale


def svg_canvas_size(
    working_w: int,
    working_h: int,
    physical_long_edge: float,
    unit: str,
    dpi: float = 96.0,
) -> tuple[float, float, float]:
    """
    Returns (svg_w_px, svg_h_px, scale_factor) given working image dims
    and desired physical longest-edge size.
    """
    long_edge = max(working_w, working_h)
    scale = to_svg_px(physical_long_edge, unit, dpi) / long_edge
    return working_w * scale, working_h * scale, scale
