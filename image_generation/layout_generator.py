from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LINE_RE = re.compile(r"^\((?P<depth>\d+),\s*(?P<val_depth>\d+)\)\s+(?P<label>.+?)\s*$")
NODE_RE = re.compile(
    r"^(?P<kind>Splitter\((?:2|3)\)|Structure A|Structure B|Output|Discard)" r"(?:\s+\[v=(?P<value>[^\]]+)\])?$"
)


@dataclass
class TreeNode:
    kind: str
    depth: int
    val_depth: int
    value: str | None = None
    children: List["TreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class Template:
    width: int
    height: int
    input_port: Tuple[float, float]
    output_ports: Tuple[Tuple[float, float], ...]
    style: str = "generic"


DEFAULT_TEMPLATES: Dict[str, Template] = {
    "Splitter(2)": Template(
        width=3,
        height=1,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 0.5), (2.5, 0.5)),
        style="splitter_2",
    ),
    "Splitter(3)": Template(
        width=3,
        height=1,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 0.5), (1.5, 0.5), (2.5, 0.5)),
        style="splitter_3",
    ),
    "Structure A": Template(
        width=3,
        height=2,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 1.5), (1.5, 1.5)),
        style="structure_a",
    ),
    "Structure B": Template(
        width=3,
        height=2,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 1.5), (1.5, 1.5)),
        style="structure_b",
    ),
    "Output": Template(width=3, height=2, input_port=(1.5, 0.5), output_ports=(), style="output"),
    "Discard": Template(width=3, height=2, input_port=(1.5, 0.5), output_ports=(), style="discard"),
}


@dataclass
class PlacedNode:
    node: TreeNode
    x: float
    y: float
    width: int
    height: int
    input_port: Tuple[float, float]
    output_ports: Tuple[Tuple[float, float], ...]


def load_tree(input_path: Path) -> TreeNode:
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return tree_from_dict(payload)
    return parse_tree_text(input_path.read_text(encoding="utf-8"))


def tree_from_dict(data: dict) -> TreeNode:
    kind = data.get("label") or data["type"]
    return TreeNode(
        kind=kind,
        depth=int(data["depth"]),
        val_depth=int(data["val_depth"]),
        value=data.get("input_val"),
        children=[tree_from_dict(child) for child in data.get("children", [])],
    )


def parse_tree_text(text: str) -> TreeNode:
    parents_by_depth: Dict[int, TreeNode] = {}
    root: TreeNode | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue

        marker_index = line.find("(")
        if marker_index < 0:
            continue
        stripped = line[marker_index:]

        node = parse_tree_line(stripped)
        if node.depth == 0:
            root = node
        else:
            parent = parents_by_depth.get(node.depth - 1)
            if parent is None:
                raise ValueError(f"Missing parent for node at depth {node.depth}: {line}")
            parent.children.append(node)

        parents_by_depth[node.depth] = node
        stale_depths = [depth for depth in parents_by_depth if depth > node.depth]
        for depth in stale_depths:
            del parents_by_depth[depth]

    if root is None:
        raise ValueError("No tree nodes found in input text.")

    return root


def parse_tree_line(line: str) -> TreeNode:
    match = LINE_RE.match(line)
    if not match:
        raise ValueError(f"Invalid tree line: {line}")

    label = match.group("label")
    node_match = NODE_RE.match(label)
    if not node_match:
        raise ValueError(f"Unsupported node label: {label}")

    return TreeNode(
        kind=node_match.group("kind"),
        depth=int(match.group("depth")),
        val_depth=int(match.group("val_depth")),
        value=node_match.group("value"),
    )


def build_schematic_positions(
    node: TreeNode,
    positions: Dict[int, PlacedNode],
    templates: Dict[str, Template],
    next_leaf_y: List[int],
    x_gap: int = 180,
    y_gap: int = 110,
) -> int:
    template = templates[node.kind]

    if not node.children:
        center_y = next_leaf_y[0]
        next_leaf_y[0] += y_gap
    else:
        child_centers = [
            build_schematic_positions(child, positions, templates, next_leaf_y, x_gap, y_gap) for child in node.children
        ]
        center_y = sum(child_centers) // len(child_centers)

    top_y = center_y - (template.height * 20) // 2
    positions[id(node)] = PlacedNode(
        node=node,
        x=node.depth * x_gap,
        y=top_y,
        width=template.width * 20,
        height=template.height * 20,
        input_port=template.input_port,
        output_ports=template.output_ports,
    )
    return center_y


TREE_GAP_X = 1
TREE_GAP_Y = 2
CELL_SIZE = 50
CANVAS_MARGIN = 1
SHOW_TEMPLATE_BOUNDS = False
MACHINE_IMAGE_HREFS: Dict[str, str] = {
    "splitter_down": "icons/splitter_down.png",
    "splitter_left": "icons/splitter_left.png",
    "merger_down": "icons/merger_down.png",
}


def measure_subtree_width(node: TreeNode, templates: Dict[str, Template]) -> int:
    template = templates[node.kind]
    if not node.children:
        return template.width

    child_widths = [measure_subtree_width(child, templates) for child in node.children]
    combined_width = sum(child_widths) + TREE_GAP_X * (len(child_widths) - 1)
    return max(template.width, combined_width)


def build_placement_positions(
    node: TreeNode,
    positions: Dict[int, PlacedNode],
    templates: Dict[str, Template],
    left: int,
    top: int,
) -> int:
    template = templates[node.kind]
    subtree_width = measure_subtree_width(node, templates)

    node_x = left + (subtree_width - template.width) // 2
    positions[id(node)] = PlacedNode(
        node=node,
        x=node_x,
        y=top,
        width=template.width,
        height=template.height,
        input_port=template.input_port,
        output_ports=template.output_ports,
    )

    if node.children:
        child_widths = [measure_subtree_width(child, templates) for child in node.children]
        children_total = sum(child_widths) + TREE_GAP_X * (len(child_widths) - 1)
        child_left = left + (subtree_width - children_total) // 2
        child_top = top + template.height + TREE_GAP_Y
        for child, width in zip(node.children, child_widths):
            build_placement_positions(child, positions, templates, child_left, child_top)
            child_left += width + TREE_GAP_X

    return subtree_width


def iter_edges(root: TreeNode) -> Iterable[Tuple[TreeNode, int, TreeNode]]:
    for parent in walk_tree(root):
        for index, child in enumerate(parent.children):
            yield parent, index, child


def walk_tree(node: TreeNode) -> Iterable[TreeNode]:
    yield node
    for child in node.children:
        yield from walk_tree(child)


def node_fill(kind: str) -> str:
    return {
        "Splitter(2)": "#cfe8ff",
        "Splitter(3)": "#bfe3cf",
        "Structure A": "#ffe6a7",
        "Structure B": "#ffd1b3",
        "Output": "#d7f9d1",
        "Discard": "#f2d7d5",
    }[kind]


def render_schematic_svg(root: TreeNode, templates: Dict[str, Template]) -> str:
    positions: Dict[int, PlacedNode] = {}
    build_schematic_positions(root, positions, templates, next_leaf_y=[80])

    placed_nodes = list(positions.values())
    max_x = max(node.x + node.width for node in placed_nodes) + 80
    max_y = max(node.y + node.height for node in placed_nodes) + 80

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_x}" height="{max_y}" viewBox="0 0 {max_x} {max_y}">',
        "<style>",
        'text { font-family: Consolas, "Courier New", monospace; fill: #222; }',
        ".edge { stroke: #444; stroke-width: 3; fill: none; }",
        ".box { stroke: #222; stroke-width: 2; rx: 10; ry: 10; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#faf8f3"/>',
    ]

    for parent, child_index, child in iter_edges(root):
        parent_box = positions[id(parent)]
        child_box = positions[id(child)]
        out_port = parent_box.output_ports[child_index]

        start_x = parent_box.x + out_port[0] * 20
        start_y = parent_box.y + out_port[1] * 20
        end_x = child_box.x + child_box.input_port[0] * 20
        end_y = child_box.y + child_box.input_port[1] * 20
        mid_x = (start_x + end_x) // 2
        path = f"M {start_x} {start_y} L {mid_x} {start_y} L {mid_x} {end_y} L {end_x} {end_y}"
        parts.append(f'<path class="edge" d="{path}" />')

    for placed in placed_nodes:
        parts.append(
            f'<rect class="box" x="{placed.x}" y="{placed.y}" width="{placed.width}" height="{placed.height}" '
            f'fill="{node_fill(placed.node.kind)}" />'
        )
        parts.append(
            f'<text x="{placed.x + 10}" y="{placed.y + 24}" font-size="15">{escape_xml(placed.node.kind)}</text>'
        )
        if placed.node.value is not None:
            parts.append(
                f'<text x="{placed.x + 10}" y="{placed.y + 42}" font-size="13">v={escape_xml(placed.node.value)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def render_svg(root: TreeNode, templates: Dict[str, Template]) -> str:
    return render_placement_svg(root, templates)


def render_placement_svg(root: TreeNode, templates: Dict[str, Template]) -> str:
    positions: Dict[int, PlacedNode] = {}
    total_width = build_placement_positions(root, positions, templates, left=CANVAS_MARGIN, top=CANVAS_MARGIN)

    placed_nodes = list(positions.values())
    max_width = max(node.x + node.width for node in placed_nodes) + CANVAS_MARGIN
    max_height = max(node.y + node.height for node in placed_nodes) + CANVAS_MARGIN
    width_px = int(max_width * CELL_SIZE)
    height_px = int(max_height * CELL_SIZE)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="0 0 {width_px} {height_px}">',
        "<style>",
        'text { font-family: Consolas, "Courier New", monospace; fill: #222; }',
        ".grid { stroke: #cfcfcf; stroke-width: 1; opacity: 0.9; }",
        ".route { stroke: rgb(245, 252, 142); stroke-width: 24; fill: none; stroke-linecap: round; stroke-linejoin: round; }",
        ".route-shadow { stroke: rgb(190, 190, 190); stroke-width: 30; fill: none; stroke-linecap: round; stroke-linejoin: round; opacity: 0.8; }",
        ".machine { fill: #efefef; stroke: #222; stroke-width: 3; rx: 8; ry: 8; }",
        ".terminal-output { fill: #dff4d6; stroke: #2d4b2f; stroke-width: 2; }",
        ".terminal-discard { fill: #f8d9d4; stroke: #66342e; stroke-width: 2; }",
        ".label { font-size: 14px; }",
        ".belt-dir { fill: #222; opacity: 0.9; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#f6f4ef"/>',
    ]

    draw_grid(parts, width_px, height_px)
    append_root_input_route(parts, positions[id(root)])

    for parent, child_index, child in iter_edges(root):
        parent_box = positions[id(parent)]
        child_box = positions[id(child)]
        start = absolute_port(parent_box, parent_box.output_ports[child_index])
        end = absolute_port(child_box, child_box.input_port)
        append_route(parts, parent_box, child_index, start, end)

    for placed in placed_nodes:
        render_node(parts, placed, templates[placed.node.kind], placed.node.value)

    parts.append("</svg>")
    return "\n".join(parts)


def draw_grid(parts: List[str], width_px: int, height_px: int) -> None:
    step = CELL_SIZE
    for x in range(0, width_px + 1, step):
        parts.append(f'<line class="grid" x1="{x}" y1="0" x2="{x}" y2="{height_px}" />')
    for y in range(0, height_px + 1, step):
        parts.append(f'<line class="grid" x1="0" y1="{y}" x2="{width_px}" y2="{y}" />')


def absolute_port(placed: PlacedNode, port: Tuple[float, float]) -> Tuple[float, float]:
    return ((placed.x + port[0]) * CELL_SIZE, (placed.y + port[1]) * CELL_SIZE)


def append_root_input_route(parts: List[str], root_box: PlacedNode) -> None:
    input_x = root_box.x + root_box.input_port[0]
    input_y = root_box.y + root_box.input_port[1]
    belt_path_absolute(parts, [(input_x, input_y - 1.0), (input_x, input_y)])


def append_route(
    parts: List[str],
    parent_box: PlacedNode,
    child_index: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    route_points = build_route_points(parent_box, child_index, start, end)
    assert_pixel_route_on_grid(route_points)
    path = "M " + " L ".join(f"{x} {y}" for x, y in route_points)
    parts.append(f'<path class="route-shadow" d="{path}" />')
    parts.append(f'<path class="route" d="{path}" />')
    append_direction_arrows(parts, [(x / CELL_SIZE, y / CELL_SIZE) for x, y in route_points])


def build_route_points(
    parent_box: PlacedNode,
    child_index: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> List[Tuple[float, float]]:
    points = [start]
    route_y = start[1] + CELL_SIZE
    points.append((start[0], route_y))

    # The center output of Splitter(3) can overlap the left/right lanes near the node.
    # Drop it one extra cell before routing across to keep the three exits separated.
    if parent_box.node.kind == "Splitter(3)" and child_index == 1:
        route_y += CELL_SIZE
        points.append((start[0], route_y))

    points.extend([(end[0], route_y), end])
    return dedupe_consecutive_points(points)


def dedupe_consecutive_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []

    deduped = [points[0]]
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)
    return deduped


def render_node(parts: List[str], placed: PlacedNode, template: Template, value: str | None) -> None:
    if SHOW_TEMPLATE_BOUNDS:
        draw_template_bounds(parts, placed)
    match template.style:
        case "structure_a":
            render_structure_a(parts, placed, value)
        case "structure_b":
            render_structure_b(parts, placed, value)
        case "splitter_2":
            render_splitter(parts, placed, value, 2)
        case "splitter_3":
            render_splitter(parts, placed, value, 3)
        case "output":
            render_terminal(parts, placed, value, is_output=True)
        case "discard":
            render_terminal(parts, placed, value, is_output=False)
        case _:
            render_generic_box(parts, placed, value)


def render_structure_a(parts: List[str], placed: PlacedNode, value: str | None) -> None:
    belt_path(parts, placed, [(1.5, 0.5), (1.5, 1.5)])
    belt_path(parts, placed, [(1.5, 0.5), (0.5, 0.5), (0.5, 1.5)])
    belt_path(parts, placed, [(1.5, 0.5), (2.5, 0.5), (2.5, 1.5), (1.5, 1.5)])
    draw_machine(parts, placed, 1.0, 0.0, machine_image_href("splitter_down"))
    draw_machine(parts, placed, 1.0, 1.0, machine_image_href("merger_down"))
    draw_value_label(parts, placed, "A", value)


def render_structure_b(parts: List[str], placed: PlacedNode, value: str | None) -> None:
    belt_path(parts, placed, [(1.5, 0.5), (1.5, 1.5)])
    belt_path(parts, placed, [(1.5, 0.5), (0.5, 0.5), (0.5, 1.5)])
    belt_path(parts, placed, [(1.5, 1.5), (0.5, 1.5)])
    draw_machine(parts, placed, 0.0, 1.0, machine_image_href("splitter_down"))
    draw_machine(parts, placed, 1.0, 0.0, machine_image_href("merger_down"))
    draw_machine(parts, placed, 1.0, 1.0, machine_image_href("splitter_left"))
    draw_value_label(parts, placed, "B", value)


def render_splitter(parts: List[str], placed: PlacedNode, value: str | None, outputs: int) -> None:
    belt_path(parts, placed, [(1.5, 0.5), (0.5, 0.5)])
    belt_path(parts, placed, [(1.5, 0.5), (2.5, 0.5)])
    draw_machine(parts, placed, 1.0, 0.0, machine_image_href("splitter_down"))
    draw_value_label(parts, placed, f"S{outputs}", value)


def render_terminal(parts: List[str], placed: PlacedNode, value: str | None, is_output: bool) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    klass = "terminal-output" if is_output else "terminal-discard"
    title = "Output" if is_output else "Discard"
    parts.append(
        f'<rect class="{klass}" x="{x + 8}" y="{y + 8}" width="{width - 16}" height="{height - 16}" rx="12" ry="12" />'
    )
    center_x = x + width / 2
    center_y = y + height / 2
    if value is not None and is_output:
        parts.append(
            f'<text class="label" x="{center_x}" y="{center_y - 9}" text-anchor="middle" dominant-baseline="middle">{title}</text>'
        )
        parts.append(
            f'<text class="label" x="{center_x}" y="{center_y + 9}" text-anchor="middle" dominant-baseline="middle">v={escape_xml(value)}</text>'
        )
    else:
        parts.append(
            f'<text class="label" x="{center_x}" y="{center_y}" text-anchor="middle" dominant-baseline="middle">{title}</text>'
        )


def render_generic_box(parts: List[str], placed: PlacedNode, value: str | None) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    parts.append(f'<rect class="machine" x="{x + 8}" y="{y + 8}" width="{width - 16}" height="{height - 16}" />')
    draw_value_label(parts, placed, placed.node.kind, value)


def draw_template_bounds(parts: List[str], placed: PlacedNode) -> None:
    x = placed.x * CELL_SIZE
    y = placed.y * CELL_SIZE
    width = placed.width * CELL_SIZE
    height = placed.height * CELL_SIZE
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="none" stroke="#d14" stroke-width="2" stroke-dasharray="8 6" opacity="0.85" />'
    )


def machine_image_href(style: str) -> str | None:
    return MACHINE_IMAGE_HREFS.get(style)


def draw_machine(
    parts: List[str], placed: PlacedNode, cell_x: float, cell_y: float, image_href: str | None = None
) -> None:
    draw_machine_absolute(parts, placed.x + cell_x, placed.y + cell_y, image_href)


def draw_machine_absolute(parts: List[str], cell_x: float, cell_y: float, image_href: str | None = None) -> None:
    if int(cell_x) != cell_x or int(cell_y) != cell_y:
        raise ValueError(f"Machine boxes must align to the grid, got ({cell_x}, {cell_y})")
    x = cell_x * CELL_SIZE
    y = cell_y * CELL_SIZE
    size = CELL_SIZE
    if image_href:
        parts.append(
            f'<image href="{escape_xml(image_href)}" x="{x + 3}" y="{y + 3}" '
            f'width="{size - 6}" height="{size - 6}" preserveAspectRatio="xMidYMid meet" />'
        )


def belt_path(parts: List[str], placed: PlacedNode, points: List[Tuple[float, float]]) -> None:
    absolute_points = [(placed.x + x, placed.y + y) for x, y in points]
    belt_path_absolute(parts, absolute_points)


def belt_path_absolute(parts: List[str], points: List[Tuple[float, float]]) -> None:
    assert_cell_route_on_grid(points)
    path = " M ".join("")  # keeps mypy quiet in older tooling
    path = "M " + " L ".join(f"{x * CELL_SIZE} {y * CELL_SIZE}" for x, y in points)
    parts.append(f'<path class="route-shadow" d="{path}" />')
    parts.append(f'<path class="route" d="{path}" />')
    append_direction_arrows(parts, points)


def snap_pixel_to_half_grid(value: float) -> float:
    cell_value = value / CELL_SIZE
    centerline_value = round(cell_value - 0.5) + 0.5
    return centerline_value * CELL_SIZE


def is_grid_coord(value: float) -> bool:
    return is_half_integer(value)


def assert_cell_route_on_grid(points: List[Tuple[float, float]]) -> None:
    if not points:
        return

    for x, y in points:
        if not is_grid_coord(x) or not is_grid_coord(y):
            raise ValueError(f"Conveyor path must stay on cell centerlines, got ({x}, {y})")

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if x1 != x2 and y1 != y2:
            raise ValueError(f"Conveyor path must be orthogonal, got segment ({x1}, {y1}) -> ({x2}, {y2})")


def assert_pixel_route_on_grid(points: List[Tuple[float, float]]) -> None:
    cell_points = [(x / CELL_SIZE, y / CELL_SIZE) for x, y in points]
    assert_cell_route_on_grid(cell_points)


def is_cell_center(x: float, y: float) -> bool:
    return is_half_integer(x) and is_half_integer(y)


def is_half_integer(value: float) -> bool:
    return abs((value * 2) - round(value * 2)) < 1e-9 and int(round(value * 2)) % 2 == 1


def append_direction_arrows(parts: List[str], points: List[Tuple[float, float]]) -> None:
    if len(points) < 2:
        return

    arrow_cells: Dict[Tuple[float, float], str] = {}

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        dy = y2 - y1
        if dx and dy:
            continue

        if dx:
            steps = int(round(abs(dx)))
            if steps <= 0:
                continue
            arrow = "right" if dx > 0 else "left"
            step_x = 1 if dx > 0 else -1
            for i in range(1, steps + 1):
                cell_x = x1 + i * step_x
                cell_y = y1
                if is_cell_center(cell_x, cell_y):
                    arrow_cells[(cell_x, cell_y)] = arrow
        elif dy:
            steps = int(round(abs(dy)))
            if steps <= 0:
                continue
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

    for (cell_x, cell_y), arrow in arrow_cells.items():
        append_arrow(parts, cell_x, cell_y, arrow)


def append_arrow(parts: List[str], cell_x: float, cell_y: float, arrow: str) -> None:
    x = cell_x * CELL_SIZE
    y = cell_y * CELL_SIZE
    rotation = {
        "right": 0,
        "down_right": 45,
        "down": 90,
        "down_left": 135,
        "left": 180,
        "up_left": 225,
        "up": 270,
        "up_right": 315,
    }[arrow]
    points = f"{x + 5},{y} {x - 3.5},{y - 3.5} {x - 3.5},{y + 3.5}"
    parts.append(f'<polygon class="belt-dir" points="{points}" transform="rotate({rotation} {x} {y})" />')


def corner_arrow(
    prev_point: Tuple[float, float], turn_point: Tuple[float, float], next_point: Tuple[float, float]
) -> str | None:
    in_dx = turn_point[0] - prev_point[0]
    in_dy = turn_point[1] - prev_point[1]
    out_dx = next_point[0] - turn_point[0]
    out_dy = next_point[1] - turn_point[1]

    if (in_dx and in_dy) or (out_dx and out_dy) or (in_dx == 0 and out_dx == 0) or (in_dy == 0 and out_dy == 0):
        return None

    horizontal = in_dx if in_dx else out_dx
    vertical = in_dy if in_dy else out_dy
    if horizontal > 0 and vertical > 0:
        return "down_right"
    if horizontal > 0 and vertical < 0:
        return "up_right"
    if horizontal < 0 and vertical > 0:
        return "down_left"
    if horizontal < 0 and vertical < 0:
        return "up_left"
    return None


def draw_value_label(parts: List[str], placed: PlacedNode, name: str, value: str | None) -> None:
    x = (placed.x + placed.input_port[0] - 1.0) * CELL_SIZE
    first_line_y = (placed.y + placed.input_port[1] - 1.0) * CELL_SIZE - 8
    first_line_y = max(first_line_y, 14)
    parts.append(f'<text class="label" x="{x}" y="{first_line_y}" text-anchor="middle">{escape_xml(name)}</text>')
    if value is not None:
        second_line_y = first_line_y + 16
        parts.append(
            f'<text class="label" x="{x}" y="{second_line_y}" text-anchor="middle">v={escape_xml(value)}</text>'
        )


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a conveyor topology tree into a schematic SVG.")
    parser.add_argument("input", type=Path, help="Path to a tree text file or exported JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("layout.svg"),
        help="Path of the generated SVG file.",
    )
    parser.add_argument(
        "--style",
        choices=("placement", "schematic"),
        default="placement",
        help="Render a top-down placement view or the older abstract schematic view.",
    )
    args = parser.parse_args()

    root = load_tree(args.input)
    if args.style == "schematic":
        svg = render_schematic_svg(root, DEFAULT_TEMPLATES)
    else:
        svg = render_placement_svg(root, DEFAULT_TEMPLATES)
    args.output.write_text(svg, encoding="utf-8")
    print(f"SVG written to {args.output}")
    print("Structure A/B are now rendered from fixed templates inferred from the supplied reference image.")


if __name__ == "__main__":
    main()
