import random
from typing import List, Tuple, Dict, Set
from fractions import Fraction
import enum

from functools import lru_cache


class NodeType(enum.Enum):
    SPLITTER_2 = enum.auto()
    SPLITTER_3 = enum.auto()
    STRUCTURE_B = enum.auto()
    STRUCTURE_A = enum.auto()
    OUTPUT = enum.auto()
    DISCARD = enum.auto()

class Node(object):
    def __init__(self, children: List["Node"], input_val: Fraction, type: NodeType, depth: int) -> None:
        self.children = children
        self.input_val = input_val
        self.type = type
        self.depth = depth

        self.cost   = None
        self.output = None

        self.val_depth = get_val_depth(self.input_val)
    
    def __str__(self):
        return self.format(0)
    
    def __repr__(self):
        match self.type:
            case NodeType.SPLITTER_2:
                out_str = 'S2'
            case NodeType.SPLITTER_3:
                out_str = 'S3'
            case NodeType.STRUCTURE_B:
                out_str = 'SA'
            case NodeType.STRUCTURE_A:
                out_str = 'SB'
            case NodeType.OUTPUT:
                out_str = 'O'
            case NodeType.DISCARD:
                out_str = 'D'
        
        if self.children:
            children_str = [repr(child) for child in self.children] 
            return out_str + '(%s)' % ''.join(children_str)
        else:
            return out_str

    def get_cost(self):
        if self.cost is None:
            match self.type:
                case NodeType.SPLITTER_2:
                    cost = 1
                case NodeType.SPLITTER_3:
                    cost = 1
                case NodeType.STRUCTURE_A:
                    cost = 1
                case NodeType.STRUCTURE_B:
                    cost = 2
                case NodeType.OUTPUT | NodeType.DISCARD:
                    cost = 0
            
            children_cost = sum(child.get_cost() for child in self.children)

            self.cost = cost + children_cost

        return self.cost
    
    def get_output(self):
        if self.output is None:
            match self.type:
                case NodeType.OUTPUT:
                    self.output = self.input_val
                case _:
                    self.output = sum(child.get_output() for child in self.children)
        
        return self.output
        
    
    def get_val_depth(self):
        return self.val_depth
    
    def mutate(self, max_depth: int = 5):
        nodes = self.collect_nodes_by_path()[1:]
        
        chosen_path, _ = random.choice(nodes)
        chosen_node = self.get_node_by_path(chosen_path)
        if chosen_node is None:
            return
        
        parent_path = chosen_path[:-1]
        parent_node = self.get_node_by_path(parent_path)
        if parent_node is None:
            return

        # type_used = set()
        # for i, child in enumerate(parent_node.children):
        #     if i != chosen_path[-1]:
        #         type_used.add(child.type)

        new_node = Node.create_random_tree(
            chosen_node.input_val,
            chosen_node.depth,
            max_depth,
            parent_node.type, 
            # type_used            
        )

        parent_node.children[chosen_path[-1]] = new_node
    
    def update_input_val(self, input_val: Fraction) -> None:
        self.input_val = input_val
        match self.type:
            case NodeType.SPLITTER_2:
                for child in self.children:
                    child.update_input_val(self.input_val / 2)
            case NodeType.SPLITTER_3:
                for child in self.children:
                    child.update_input_val(self.input_val / 3)
            case NodeType.STRUCTURE_B:
                self.children[0].update_input_val(self.input_val * Fraction(3, 4))
                self.children[1].update_input_val(self.input_val * Fraction(1, 4))
            case NodeType.STRUCTURE_A:
                self.children[0].update_input_val(self.input_val * Fraction(2, 3))
                self.children[1].update_input_val(self.input_val * Fraction(1, 3))
            case NodeType.OUTPUT | NodeType.DISCARD:
                return

    def deep_copy(self, parent_depth: int = -1) -> "Node":
        if parent_depth >= 0:
            current_depth = parent_depth + 1
        else:
            current_depth = self.depth
        
        new_children = []
        for child in self.children:
            new_child = child.deep_copy(current_depth)
            new_children.append(new_child)
        
        return Node(
            children=new_children,
            input_val=self.input_val,
            type=self.type,
            depth=current_depth
        )
    
    def collect_nodes_by_path(self, current_path: List[int] | None = None) -> List[Tuple[List[int], int]]:
        if current_path is None:
            current_path = []
        
        result = []
        
        result.append((current_path.copy(), self.get_val_depth()))
        
        for i, child in enumerate(self.children):
            child_path = current_path + [i]
            result.extend(child.collect_nodes_by_path(child_path))
        
        return result
    
    def get_node_by_path(self, path: List[int]) -> "Node | None":
        current_node = self
        
        for index in path:
            if 0 <= index < len(current_node.children):
                current_node = current_node.children[index]
            else:
                return None
        
        return current_node
    
    def replace_node(self, path: List[int], new_node: "Node") -> bool:
        if not path:
            return False
        
        parent_path = path[:-1]
        child_index = path[-1]
        
        parent_node = self.get_node_by_path(parent_path)
        if parent_node is None or child_index < 0 or child_index >= len(parent_node.children):
            return False
        
        child = parent_node.children[child_index]
        new_node.update_input_val(child.input_val)
        parent_node.children[child_index] = new_node

        return True

    def to_dict(self) -> dict:
        return {
            "type": self.type.name,
            "label": self.get_display_name(),
            "input_val": str(self.input_val),
            "input_num": self.input_val.numerator,
            "input_den": self.input_val.denominator,
            "depth": self.depth,
            "val_depth": self.get_val_depth(),
            "children": [child.to_dict() for child in self.children],
        }

    def get_display_name(self) -> str:
        match self.type:
            case NodeType.SPLITTER_2:
                return "Splitter(2)"
            case NodeType.SPLITTER_3:
                return "Splitter(3)"
            case NodeType.STRUCTURE_B:
                return "Structure B"
            case NodeType.STRUCTURE_A:
                return "Structure A"
            case NodeType.OUTPUT:
                return "Output"
            case NodeType.DISCARD:
                return "Discard"
        raise NotImplementedError()

    @staticmethod
    def create_random_tree(input_val: Fraction = Fraction(1, 1), 
                        depth: int = 0, max_depth: int = 5, 
                        current_type: None | NodeType = None, 
                        type_used: Set[NodeType] = set()) -> "Node":
        
        val_depth = get_val_depth(input_val)

        if depth >= max_depth or val_depth >= max_depth:
            c = {4, 5}
        else:
            match current_type:
                case None:
                    c = {0, 1, 2, 3}
                case NodeType.SPLITTER_2 | NodeType.SPLITTER_3:
                    c = {0, 1, 2, 3, 4, 5}
                case NodeType.STRUCTURE_B:
                    c = {0, 2, 3, 4, 5}
                case NodeType.STRUCTURE_A:
                    c = {1, 2, 3, 4, 5}
                case _:
                    raise NotImplementedError()
                
        if depth + 1 >= max_depth or val_depth + 1 >= max_depth:
            c.discard(1)
            if val_depth + 1 >= max_depth:
                c.discard(2)

        if NodeType.OUTPUT in type_used:
            c.discard(4)
        if NodeType.DISCARD in type_used:
            c.discard(5)
        
        try:
            idx = random.choice(tuple(c))
        except IndexError as e:
            print(input_val, depth, current_type, type_used)
            raise e

        match idx:
            case 0:
                child_1 = Node.create_random_tree(
                    input_val / 2, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.SPLITTER_2
                )
                child_2 = Node.create_random_tree(
                    input_val / 2, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.SPLITTER_2, 
                    type_used={child_1.type}
                )
                return Node(
                    [child_1, child_2], 
                    input_val, 
                    NodeType.SPLITTER_2, 
                    depth
                )
            case 1:
                child_1 = Node.create_random_tree(
                    input_val / 3, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.SPLITTER_3
                )
                child_2 = Node.create_random_tree(
                    input_val / 3, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.SPLITTER_3, 
                    type_used={child_1.type}
                )
                child_3 = Node.create_random_tree(
                    input_val / 3, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.SPLITTER_3, 
                    type_used={child_1.type, child_2.type}
                )
                return Node(
                    [child_1, child_2, child_3], 
                    input_val, 
                    NodeType.SPLITTER_3, 
                    depth
                )
            case 2:
                child_1 = Node.create_random_tree(
                    input_val * Fraction(3, 4), 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.STRUCTURE_B
                )
                child_2 = Node.create_random_tree(
                    input_val * Fraction(1, 4), 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.STRUCTURE_B, 
                    type_used={child_1.type}
                )
                return Node(
                    [child_1, child_2], 
                    input_val, 
                    NodeType.STRUCTURE_B, 
                    depth
                )
            case 3:
                child_1 = Node.create_random_tree(
                    input_val * Fraction(2, 3), 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.STRUCTURE_A
                )
                child_2 = Node.create_random_tree(
                    input_val * Fraction(1, 3),     
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    current_type=NodeType.STRUCTURE_A, 
                    type_used={child_1.type}
                )
                return Node(
                    [child_1, child_2], 
                    input_val, 
                    NodeType.STRUCTURE_A, 
                    depth
                )
            case 4:
                return Node(
                    [], 
                    input_val, 
                    NodeType.OUTPUT, 
                    depth
                )
            case 5:
                return Node(
                    [], 
                    input_val, 
                    NodeType.DISCARD, 
                    depth
                )
            case _:
                raise NotImplementedError()

    # @staticmethod
    # def replace_node_in_tree(root: "Node", target_original: "Node", replacement: "Node") -> "Node":
    #     if root is target_original:
    #         return replacement
        
    #     for i, child in enumerate(root.children):
    #         if child is target_original:
    #             root.children[i] = replacement
    #             return root
    #         else:
    #             new_child = Node.replace_node_in_tree(child, target_original, replacement)
    #             if new_child is not child:
    #                 root.children[i] = new_child
    #                 return root
        
    #     return root

    @staticmethod
    def crossover_trees(tree_1: "Node", tree_2: "Node") -> "Tuple[Node, Node]":
        # deep copy
        new_tree_1 = tree_1.deep_copy()
        new_tree_2 = tree_2.deep_copy()
        
        # collect nodes
        nodes_1_info = new_tree_1.collect_nodes_by_path()
        nodes_2_info = new_tree_2.collect_nodes_by_path()
        
        # build dict
        depth_map_1: Dict[int, List[Tuple[List[int], Node]]] = {}
        depth_map_2: Dict[int, List[Tuple[List[int], Node]]] = {}
        
        for path, val_depth in nodes_1_info:
            node = new_tree_1.get_node_by_path(path)
            if node is not None and node.type not in [NodeType.OUTPUT, NodeType.DISCARD]:
                depth_map_1.setdefault(val_depth, []).append((path, node))
        
        for path, val_depth in nodes_2_info:
            node = new_tree_2.get_node_by_path(path)
            if node is not None and node.type not in [NodeType.OUTPUT, NodeType.DISCARD]:
                depth_map_2.setdefault(val_depth, []).append((path, node))
        
        # get common depths
        common_depths = set(depth_map_1.keys()) & set(depth_map_2.keys())
        
        if not common_depths:
            return new_tree_1, new_tree_2
        
        chosen_depth = random.choice(list(common_depths))
        
        # choice target
        path_1, node_1 = random.choice(depth_map_1[chosen_depth])
        path_2, node_2 = random.choice(depth_map_2[chosen_depth])
        
        node_1_copy = node_1.deep_copy()
        node_2_copy = node_2.deep_copy()
                
        # replace
        success_1 = new_tree_1.replace_node(path_1, node_2_copy)
        success_2 = new_tree_2.replace_node(path_2, node_1_copy)
        
        if not success_1 or not success_2:
            return tree_1.deep_copy(), tree_2.deep_copy()
        
        return new_tree_1, new_tree_2

    @staticmethod
    def is_valid(tree: "Node") -> bool:
        # SPLITTER_2 = enum.auto()
        # SPLITTER_3 = enum.auto()
        # STRUCTURE_B = enum.auto()
        # STRUCTURE_A = enum.auto()
        # OUTPUT = enum.auto()
        # DISCARD = enum.auto()
        output_n = sum(child.type == NodeType.OUTPUT for child in tree.children)
        discard_n = sum(child.type == NodeType.DISCARD for child in tree.children)
        if output_n > 1 or discard_n > 1:
            return False
        
        match tree.type:
            case NodeType.SPLITTER_2:
                if len(tree.children) != 2:
                    return False
                
                return all(Node.is_valid(child) for child in tree.children)
            case NodeType.SPLITTER_3:
                if len(tree.children) != 3:
                    return False
                
                return all(Node.is_valid(child) for child in tree.children)
                
            case NodeType.STRUCTURE_B:
                if any(child.type == NodeType.SPLITTER_3 for child in tree.children):
                    return False
                
                return all(Node.is_valid(child) for child in tree.children)
            
            case NodeType.STRUCTURE_A:
                if any(child.type == NodeType.SPLITTER_2 for child in tree.children):
                    return False
                
                return all(Node.is_valid(child) for child in tree.children)
            
            case NodeType.OUTPUT | NodeType.DISCARD:
                if len(tree.children) != 0:
                    return False
                
                return True
            
            case _:
                raise NotImplementedError()

    def to_string(self) -> str:
        return self.__repr__()

    def format(self, indent: int) -> str:
        label = self.get_display_name()
        if self.type == NodeType.DISCARD:
            out_str = '  ' * self.depth + label
        else:
            out_str = '  ' * self.depth + '%s [v=%s]' % (label, self.input_val)
        
        children_str = [child.format(indent) for child in self.children]

        return ('\n' + ' ' * indent).join(['(%d, %d) ' % (self.depth, self.get_val_depth()) + out_str, *children_str])

@lru_cache()
def get_val_depth(input_val: Fraction) -> int:
    denominator = input_val.denominator
    m = 0
    while denominator % 2 == 0:
        denominator //= 2
        m += 1
    n = 0
    while denominator % 3 == 0:
        denominator //= 3
        n += 1

    return m + n

if __name__ == '__main__':
    tree_1 = Node.create_random_tree(max_depth=3)
    tree_2 = Node.create_random_tree(max_depth=3)
    for _ in range(10):
        tree_1, tree_2 = Node.crossover_trees(tree_1, tree_2)
        print(tree_1)
        print()
        print(tree_2)
        print('=' * 60)
    
