from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from layout_generator import (
    CANVAS_MARGIN,
    CELL_SIZE,
    DEFAULT_TEMPLATES,
    MACHINE_IMAGE_HREFS,
    PlacedNode,
    Template,
    TreeNode,
    absolute_port,
    build_placement_positions,
    build_route_points,
    corner_arrow,
    is_cell_center,
    iter_edges,
    tree_from_dict,
)

BACKGROUND_COLOR = (246, 244, 239, 255)
GRID_COLOR = (207, 207, 207, 230)
ROUTE_COLOR = (245, 252, 142, 255)
ROUTE_SHADOW_COLOR = (190, 190, 190, 204)
TEXT_COLOR = (34, 34, 34, 255)
OUTPUT_FILL = (223, 244, 214, 255)
OUTPUT_OUTLINE = (45, 75, 47, 255)
DISCARD_FILL = (248, 217, 212, 255)
DISCARD_OUTLINE = (102, 52, 46, 255)
ARROW_COLOR = (34, 34, 34, 230)


def render_tree_preview_from_dict(
    tree_payload: dict,
    templates: Dict[str, Template] | None = None,
    asset_dir: Path | None = None,
) -> Image.Image:
    return render_tree_preview(tree_from_dict(tree_payload), templates=templates, asset_dir=asset_dir)


def render_tree_preview(
    root: TreeNode,
    templates: Dict[str, Template] | None = None,
    asset_dir: Path | None = None,
) -> Image.Image:
    active_templates = templates or DEFAULT_TEMPLATES
    positions: Dict[int, PlacedNode] = {}
    build_placement_positions(root, positions, active_templates, left=CANVAS_MARGIN, top=CANVAS_MARGIN)

    placed_nodes = list(positions.values())
    max_width = max(node.x + node.width for node in placed_nodes) + CANVAS_MARGIN
    max_height = max(node.y + node.height for node in placed_nodes) + CANVAS_MARGIN
    width_px = int(max_width * CELL_SIZE)
    height_px = int(max_height * CELL_SIZE)

    image = Image.new("RGBA", (width_px, height_px), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width_px, height_px)
    _draw_root_input_route(draw, positions[id(root)])

    for parent, child_index, child in iter_edges(root):
        parent_box = positions[id(parent)]
        child_box = positions[id(child)]
        start = absolute_port(parent_box, parent_box.output_ports[child_index])
        end = absolute_port(child_box, child_box.input_port)
        _draw_route(draw, parent_box, child_index, start, end)

    if asset_dir is not None:
        resolved_asset_dir = asset_dir
    elif getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        resolved_asset_dir = Path(sys._MEIPASS)
    else:
        resolved_asset_dir = Path(__file__).resolve().parent.parent
    for placed in placed_nodes:
        _render_node(image, draw, placed, active_templates[placed.node.kind], placed.node.value, resolved_asset_dir)

    return image


def _draw_grid(draw: ImageDraw.ImageDraw, width_px: int, height_px: int) -> None:
    for x in range(0, width_px + 1, CELL_SIZE):
        draw.line([(x, 0), (x, height_px)], fill=GRID_COLOR, width=1)
    for y in range(0, height_px + 1, CELL_SIZE):
        draw.line([(0, y), (width_px, y)], fill=GRID_COLOR, width=1)


def _draw_root_input_route(draw: ImageDraw.ImageDraw, root_box: PlacedNode) -> None:
    input_x = root_box.x + root_box.input_port[0]
    input_y = root_box.y + root_box.input_port[1]
    _draw_cell_path(draw, [(input_x, input_y - 1.0), (input_x, input_y)])


def _draw_route(
    draw: ImageDraw.ImageDraw,
    parent_box: PlacedNode,
    child_index: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    route_points = build_route_points(parent_box, child_index, start, end)
    _draw_pixel_path(draw, route_points)


def _draw_pixel_path(draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]]) -> None:
    draw.line(points, fill=ROUTE_SHADOW_COLOR, width=30, joint="curve")
    draw.line(points, fill=ROUTE_COLOR, width=24, joint="curve")
    _draw_direction_arrows(draw, [(x / CELL_SIZE, y / CELL_SIZE) for x, y in points])


def _draw_cell_path(draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]]) -> None:
    pixel_points = [(x * CELL_SIZE, y * CELL_SIZE) for x, y in points]
    _draw_pixel_path(draw, pixel_points)


def _render_node(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    placed: PlacedNode,
    template: Template,
    value: str | None,
    asset_dir: Path,
) -> None:
    match template.style:
        case "structure_a":
            _render_structure_a(image, draw, placed, value, asset_dir)
        case "structure_b":
            _render_structure_b(image, draw, placed, value, asset_dir)
        case "splitter_2":
            _render_splitter(image, draw, placed, value, 2, asset_dir)
        case "splitter_3":
            _render_splitter(image, draw, placed, value, 3, asset_dir)
        case "output":
            _render_terminal(draw, placed, value, is_output=True)
        case "discard":
            _render_terminal(draw, placed, value, is_output=False)
        case _:
            _render_generic_box(draw, placed, value)


def _render_structure_a(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    placed: PlacedNode,
    value: str | None,
    asset_dir: Path,
) -> None:
    _draw_internal_path(draw, placed, [(1.5, 0.5), (1.5, 1.5)])
    _draw_internal_path(draw, placed, [(1.5, 0.5), (0.5, 0.5), (0.5, 1.5)])
    _draw_internal_path(draw, placed, [(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)])
    _paste_machine(image, placed.x + 1.0, placed.y + 0.0, "splitter_down", asset_dir)
    _paste_machine(image, placed.x + 1.0, placed.y + 1.0, "merger_down", asset_dir)
    _draw_value_label(draw, placed, "A", value)


def _render_structure_b(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    placed: PlacedNode,
    value: str | None,
    asset_dir: Path,
) -> None:
    _draw_internal_path(draw, placed, [(1.5, 0.5), (1.5, 1.5)])
    _draw_internal_path(draw, placed, [(1.5, 0.5), (0.5, 0.5), (0.5, 1.5)])
    _draw_internal_path(draw, placed, [(1.5, 1.5), (0.5, 1.5)])
    _paste_machine(image, placed.x + 0.0, placed.y + 1.0, "splitter_down", asset_dir)
    _paste_machine(image, placed.x + 1.0, placed.y + 0.0, "merger_down", asset_dir)
    _paste_machine(image, placed.x + 1.0, placed.y + 1.0, "splitter_left", asset_dir)
    _draw_value_label(draw, placed, "B", value)


def _render_splitter(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    placed: PlacedNode,
    value: str | None,
    outputs: int,
    asset_dir: Path,
) -> None:
    _draw_internal_path(draw, placed, [(1.5, 0.5), (0.5, 0.5)])
    _draw_internal_path(draw, placed, [(1.5, 0.5), (2.5, 0.5)])
    _paste_machine(image, placed.x + 1.0, placed.y + 0.0, "splitter_down", asset_dir)
    _draw_value_label(draw, placed, f"S{outputs}", value)


def _render_terminal(draw: ImageDraw.ImageDraw, placed: PlacedNode, value: str | None, is_output: bool) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    fill = OUTPUT_FILL if is_output else DISCARD_FILL
    outline = OUTPUT_OUTLINE if is_output else DISCARD_OUTLINE
    title = "输出" if is_output else "回流"
    draw.rounded_rectangle(
        (x + 8, y + 8, x + width - 8, y + height - 8), radius=12, fill=fill, outline=outline, width=2
    )

    title_font = _load_font(18, bold=True)
    value_font = _load_font(16)
    center_x = x + width / 2
    center_y = y + height / 2
    if value is not None and is_output:
        _draw_centered_text(draw, (center_x, center_y - 12), title, title_font)
        _draw_centered_text(draw, (center_x, center_y + 12), f"v={value}", value_font)
    else:
        _draw_centered_text(draw, (center_x, center_y), title, title_font)


def _render_generic_box(draw: ImageDraw.ImageDraw, placed: PlacedNode, value: str | None) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    draw.rounded_rectangle(
        (x + 8, y + 8, x + width - 8, y + height - 8),
        radius=10,
        fill=(239, 239, 239, 255),
        outline=(34, 34, 34, 255),
        width=3,
    )
    _draw_value_label(draw, placed, placed.node.kind, value)


def _draw_internal_path(draw: ImageDraw.ImageDraw, placed: PlacedNode, points: List[Tuple[float, float]]) -> None:
    absolute_points = [(placed.x + x, placed.y + y) for x, y in points]
    _draw_cell_path(draw, absolute_points)


def _draw_value_label(draw: ImageDraw.ImageDraw, placed: PlacedNode, name: str, value: str | None) -> None:
    x = (placed.x + placed.input_port[0] - 1.0) * CELL_SIZE
    first_line_y = (placed.y + placed.input_port[1] - 1.0) * CELL_SIZE - 8
    first_line_y = max(first_line_y, 14)
    title_font = _load_font(18, bold=True)
    value_font = _load_font(16)
    _draw_centered_text(draw, (x, first_line_y), name, title_font)
    if value is not None:
        _draw_centered_text(draw, (x, first_line_y + 18), f"v={value}", value_font)


def _draw_direction_arrows(draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]]) -> None:
    for (cell_x, cell_y), arrow in _collect_direction_arrows(points).items():
        _draw_arrow(draw, cell_x, cell_y, arrow)


def _collect_direction_arrows(points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
    if len(points) < 2:
        return {}

    arrow_cells: Dict[Tuple[float, float], str] = {}
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        if dx and dy:
            continue

        if dx:
            steps = int(round(abs(dx)))
            arrow = "right" if dx > 0 else "left"
            step_x = 1 if dx > 0 else -1
            for i in range(1, steps + 1):
                cell_x = x1 + i * step_x
                cell_y = y1
                if is_cell_center(cell_x, cell_y):
                    arrow_cells[(cell_x, cell_y)] = arrow
        elif dy:
            steps = int(round(abs(dy)))
            arrow = "down" if dy > 0 else "up"
            step_y = 1 if dy > 0 else -1
            for i in range(1, steps + 1):
                cell_x = x1
                cell_y = y1 + i * step_y
                if is_cell_center(cell_x, cell_y):
                    arrow_cells[(cell_x, cell_y)] = arrow

    for prev_point, turn_point, next_point in zip(points, points[1:], points[2:]):
        turn_arrow = corner_arrow(prev_point, turn_point, next_point)
        if turn_arrow and is_cell_center(turn_point[0], turn_point[1]):
            arrow_cells[turn_point] = turn_arrow

    return arrow_cells


def _draw_arrow(draw: ImageDraw.ImageDraw, cell_x: float, cell_y: float, arrow: str) -> None:
    center_x = cell_x * CELL_SIZE
    center_y = cell_y * CELL_SIZE
    dx, dy = {
        "right": (1.0, 0.0),
        "down_right": (1.0, 1.0),
        "down": (0.0, 1.0),
        "down_left": (-1.0, 1.0),
        "left": (-1.0, 0.0),
        "up_left": (-1.0, -1.0),
        "up": (0.0, -1.0),
        "up_right": (1.0, -1.0),
    }[arrow]
    length = math.hypot(dx, dy)
    unit_x = dx / length
    unit_y = dy / length
    perp_x = -unit_y
    perp_y = unit_x

    tip = (center_x + unit_x * 8, center_y + unit_y * 8)
    base_center = (center_x - unit_x * 2, center_y - unit_y * 2)
    base_left = (base_center[0] + perp_x * 4, base_center[1] + perp_y * 4)
    base_right = (base_center[0] - perp_x * 4, base_center[1] - perp_y * 4)
    draw.polygon([tip, base_left, base_right], fill=ARROW_COLOR)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text((center[0] - text_width / 2, center[1] - text_height / 2), text, font=font, fill=TEXT_COLOR)


def _paste_machine(image: Image.Image, cell_x: float, cell_y: float, style: str, asset_dir: Path) -> None:
    relative_path = MACHINE_IMAGE_HREFS.get(style)
    if not relative_path:
        return
    icon_path = asset_dir / relative_path
    icon = _load_icon(icon_path)
    x = int(cell_x * CELL_SIZE + 3)
    y = int(cell_y * CELL_SIZE + 3)
    image.alpha_composite(icon, (x, y))


@lru_cache(maxsize=None)
def _load_icon(icon_path: Path) -> Image.Image:
    icon = Image.open(icon_path).convert("RGBA")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return icon.resize((CELL_SIZE - 6, CELL_SIZE - 6), resampling)


@lru_cache(maxsize=None)
def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/msyhbd.ttc"),
                Path("C:/Windows/Fonts/msyhbd.ttf"),
                Path("C:/Windows/Fonts/simhei.ttf"),
                Path("C:/Windows/Fonts/Dengb.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyh.ttf"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/Deng.ttf"),
        ]
    )
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/segoeuib.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
        ]
    )

    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()
