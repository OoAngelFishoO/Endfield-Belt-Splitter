from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LINE_RE = re.compile(r"^\((?P<depth>\d+),\s*(?P<val_depth>\d+)\)\s+(?P<label>.+?)\s*$")
NODE_RE = re.compile(
    r"^(?P<kind>Splitter\((?:2|3)\)|Structure A|Structure B|Output|Discard)" r"(?:\s+\[v=(?P<value>[^\]]+)\])?$"
)

# 画布与网格参数
CANVAS_MARGIN = 1  # 画布边距
TREE_GAP_X = 1  # 节点水平间距
TREE_GAP_Y = 2  # 节点垂直间距
CELL_SIZE = 50  # 单元格大小（像素）


@dataclass
class TreeNode:
    """树节点，存储节点类型、深度和子节点"""

    kind: str
    depth: int
    val_depth: int
    value: str
    children: List["TreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class Node:
    """节点模板，定义长宽、输入/输出端口、内部机器、内部传送带和渲染风格。"""

    name: str
    width: int
    height: int
    input_port: Tuple[float, float]
    output_ports: Tuple[Tuple[float, float], ...]
    internal_paths: Tuple[Tuple[Tuple[float, float], ...], ...] = ()
    machines: Tuple[Tuple[float, float, str], ...] = ()
    label: str | None = None


@dataclass
class PlacedNode:
    """已经放置到布局中的节点，包含绝对位置以及对应的节点定义。"""

    node: TreeNode
    x: float
    y: float
    width: int
    height: int
    input_port: Tuple[float, float]
    output_ports: Tuple[Tuple[float, float], ...]
    definition: Node


@dataclass(frozen=True)
class PlacementScene:
    """完整布局场景，包含所有已放置节点以及最终画布尺寸。"""

    positions: Dict[int, PlacedNode]
    placed_nodes: Tuple[PlacedNode, ...]
    width_px: int
    height_px: int


# 各节点定义
DEFAULT_NODES: Dict[str, Node] = {
    "Splitter(2)": Node(
        name="splitter_2",
        width=3,
        height=1,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 0.5), (2.5, 0.5)),
        internal_paths=(
            ((1.5, 0.5), (0.5, 0.5)),
            ((1.5, 0.5), (2.5, 0.5)),
        ),
        machines=((1.0, 0.0, "splitter_down"),),
        label="S2",
    ),
    "Splitter(3)": Node(
        name="splitter_3",
        width=3,
        height=1,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 0.5), (1.5, 0.5), (2.5, 0.5)),
        internal_paths=(
            ((1.5, 0.5), (0.5, 0.5)),
            ((1.5, 0.5), (2.5, 0.5)),
        ),
        machines=((1.0, 0.0, "splitter_down"),),
        label="S3",
    ),
    "Structure A": Node(
        name="structure_a",
        width=3,
        height=2,
        input_port=(1.5, 0.5),
        output_ports=((0.5, 1.5), (2.5, 1.5)),
        internal_paths=(
            ((1.5, 0.5), (0.5, 0.5), (0.5, 1.5)),  # 左
            ((1.5, 0.5), (1.5, 1.5), (2.5, 1.5)),  # 中
            ((1.5, 0.5), (2.5, 0.5), (2.5, 1.5)),  # 右
        ),
        machines=((1.0, 0.0, "splitter_down"), (2.0, 1.0, "merger_down")),
        label="A",
    ),
    "Structure B": Node(
        name="structure_b",
        width=4,
        height=2,
        input_port=(3.5, 0.5),
        output_ports=((0.5, 1.5), (3.5, 1.5)),
        internal_paths=(
            ((3.5, 0.5), (0.5, 0.5), (0.5, 1.5)),  # 左上
            ((3.5, 1.5), (0.5, 1.5)),  # 左下
            ((3.5, 0.5), (3.5, 1.5)),  # 右
        ),
        machines=((3.0, 0.0, "splitter_down"), (3.0, 1.0, "splitter_down"), (0.0, 1.0, "merger_down")),
        label="B",
    ),
    "Output": Node(
        name="output",
        width=3,
        height=2,
        input_port=(1.5, 0.5),
        output_ports=(),
        label="Output",
    ),
    "Discard": Node(
        name="discard",
        width=3,
        height=2,
        input_port=(1.5, 0.5),
        output_ports=(),
        label="Discard",
    ),
}

# 内部机器贴图资源
MACHINE_IMAGE_HREFS: Dict[str, str] = {
    "splitter_down": "icons/splitter_down.png",
    "splitter_left": "icons/splitter_left.png",
    "merger_down": "icons/merger_down.png",
}


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


def measure_subtree_width(node: TreeNode, node_defs: Dict[str, Node]) -> int:
    node_def = node_defs[node.kind]
    if not node.children:
        return node_def.width

    child_widths = [measure_subtree_width(child, node_defs) for child in node.children]
    combined_width = sum(child_widths) + TREE_GAP_X * (len(child_widths) - 1)
    return max(node_def.width, combined_width)


def build_placement_positions(
    node: TreeNode,
    positions: Dict[int, PlacedNode],
    node_defs: Dict[str, Node],
    left: int,
    top: int,
) -> int:
    node_def = node_defs[node.kind]
    subtree_width = measure_subtree_width(node, node_defs)

    node_x = left + (subtree_width - node_def.width) // 2
    positions[id(node)] = PlacedNode(
        node=node,
        x=node_x,
        y=top,
        width=node_def.width,
        height=node_def.height,
        input_port=node_def.input_port,
        output_ports=node_def.output_ports,
        definition=node_def,
    )

    if node.children:
        child_widths = [measure_subtree_width(child, node_defs) for child in node.children]
        children_total = sum(child_widths) + TREE_GAP_X * (len(child_widths) - 1)
        child_left = left + (subtree_width - children_total) // 2
        child_top = top + node_def.height + TREE_GAP_Y
        for child, width in zip(node.children, child_widths):
            build_placement_positions(child, positions, node_defs, child_left, child_top)
            child_left += width + TREE_GAP_X

    return subtree_width


def build_placement_scene(root: TreeNode, node_defs: Dict[str, Node]) -> PlacementScene:
    positions: Dict[int, PlacedNode] = {}
    build_placement_positions(root, positions, node_defs, left=CANVAS_MARGIN, top=CANVAS_MARGIN)

    placed_nodes = tuple(positions.values())
    max_width = max(node.x + node.width for node in placed_nodes) + CANVAS_MARGIN
    max_height = max(node.y + node.height for node in placed_nodes) + CANVAS_MARGIN
    return PlacementScene(
        positions=positions,
        placed_nodes=placed_nodes,
        width_px=int(max_width * CELL_SIZE),
        height_px=int(max_height * CELL_SIZE),
    )


def iter_edges(root: TreeNode) -> Iterable[Tuple[TreeNode, int, TreeNode]]:
    for parent in walk_tree(root):
        for index, child in enumerate(parent.children):
            yield parent, index, child


def walk_tree(node: TreeNode) -> Iterable[TreeNode]:
    yield node
    for child in node.children:
        yield from walk_tree(child)


def absolute_port(placed: PlacedNode, port: Tuple[float, float]) -> Tuple[float, float]:
    return ((placed.x + port[0]) * CELL_SIZE, (placed.y + port[1]) * CELL_SIZE)


def build_root_input_points(root_box: PlacedNode) -> List[Tuple[float, float]]:
    input_x = root_box.x + root_box.input_port[0]
    input_y = root_box.y + root_box.input_port[1]
    return [(input_x, input_y - 1.0), (input_x, input_y)]


def build_route_points(
    parent_box: PlacedNode,
    child_index: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> List[Tuple[float, float]]:
    points = [start]
    route_y = start[1] + CELL_SIZE
    points.append((start[0], route_y))

    # Keep the center branch of a 3-way splitter from overlapping the side exits.
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


def machine_image_href(style: str) -> str | None:
    return MACHINE_IMAGE_HREFS.get(style)


def terminal_title(is_output: bool) -> str:
    return "Output" if is_output else "Discard"


def build_internal_cell_paths(placed: PlacedNode) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    return tuple(
        tuple((placed.x + x, placed.y + y) for x, y in path_points) for path_points in placed.definition.internal_paths
    )


def build_machine_cells(placed: PlacedNode) -> Tuple[Tuple[float, float, str], ...]:
    return tuple(
        (placed.x + cell_x, placed.y + cell_y, machine_style)
        for cell_x, cell_y, machine_style in placed.definition.machines
    )


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


def collect_direction_arrows(points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
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

    return arrow_cells


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
