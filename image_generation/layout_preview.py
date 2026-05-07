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
    CELL_SIZE,
    DEFAULT_NODES,
    MACHINE_IMAGE_HREFS,
    Node,
    PlacedNode,
    TreeNode,
    absolute_port,
    build_internal_cell_paths,
    build_machine_cells,
    build_placement_scene,
    build_root_input_points,
    build_route_points,
    collect_direction_arrows,
    iter_edges,
    terminal_title,
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
    node_defs: Dict[str, Node] | None = None,
    asset_dir: Path | None = None,
) -> Image.Image:
    return render_tree_preview(tree_from_dict(tree_payload), node_defs=node_defs, asset_dir=asset_dir)


def render_tree_preview(
    root: TreeNode,
    node_defs: Dict[str, Node] | None = None,
    asset_dir: Path | None = None,
) -> Image.Image:
    active_node_defs = node_defs or DEFAULT_NODES
    scene = build_placement_scene(root, active_node_defs)
    positions = scene.positions

    image = Image.new("RGBA", (scene.width_px, scene.height_px), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, scene.width_px, scene.height_px)
    _draw_root_input_route(draw, positions[id(root)])

    for parent, child_index, child in iter_edges(root):
        parent_box = positions[id(parent)]
        child_box = positions[id(child)]
        start = absolute_port(parent_box, parent_box.output_ports[child_index])
        end = absolute_port(child_box, child_box.input_port)
        _draw_route(draw, parent_box, child_index, start, end)

    resolved_asset_dir = _resolve_asset_dir(asset_dir)
    for placed in scene.placed_nodes:
        _render_node(image, draw, placed, placed.node.value, resolved_asset_dir)

    return image


def _resolve_asset_dir(asset_dir: Path | None) -> Path:
    if asset_dir is not None:
        return asset_dir
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


def _draw_grid(draw: ImageDraw.ImageDraw, width_px: int, height_px: int) -> None:
    for x in range(0, width_px + 1, CELL_SIZE):
        draw.line([(x, 0), (x, height_px)], fill=GRID_COLOR, width=1)
    for y in range(0, height_px + 1, CELL_SIZE):
        draw.line([(0, y), (width_px, y)], fill=GRID_COLOR, width=1)


def _draw_root_input_route(draw: ImageDraw.ImageDraw, root_box: PlacedNode) -> None:
    _draw_cell_path(draw, build_root_input_points(root_box))


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
    value: str | None,
    asset_dir: Path,
) -> None:
    internal_paths = build_internal_cell_paths(placed)
    machine_cells = build_machine_cells(placed)
    if internal_paths or machine_cells:
        for path_points in internal_paths:
            _draw_cell_path(draw, list(path_points))
        for cell_x, cell_y, machine_style in machine_cells:
            _paste_machine(image, cell_x, cell_y, machine_style, asset_dir)
        if placed.definition.label is not None:
            _draw_value_label(draw, placed, placed.definition.label, value)
        return

    if placed.definition.name == "output":
        _render_terminal(draw, placed, value, is_output=True)
    elif placed.definition.name == "discard":
        _render_terminal(draw, placed, value, is_output=False)
    else:
        _render_generic_box(draw, placed, value)


def _render_terminal(draw: ImageDraw.ImageDraw, placed: PlacedNode, value: str | None, is_output: bool) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    fill = OUTPUT_FILL if is_output else DISCARD_FILL
    outline = OUTPUT_OUTLINE if is_output else DISCARD_OUTLINE
    title = terminal_title(is_output)
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
    for (cell_x, cell_y), arrow in collect_direction_arrows(points).items():
        _draw_arrow(draw, cell_x, cell_y, arrow)


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
