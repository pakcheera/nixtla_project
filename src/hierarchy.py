import pandas as pd
import numpy as np
from collections import defaultdict, deque

from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut
from collections import defaultdict, deque
import numpy as np
import pandas as pd


class Hierarchy:
    def __init__(self, config, freq="MS"):
        self.config = config
        self.freq = freq

        # --- read hierarchy CSV (parent-child edge list) ---
        df = pd.read_csv(config["paths"]["hierarchy_csv_path"])
        df = df.rename(columns={":parent": "parent", ":child": "child"})  # if needed

        # normalize
        df["parent"] = df["parent"].fillna("").astype(str)
        df["child"] = df["child"].fillna("").astype(str)

        # remove empty children + duplicate edges
        df = df[df["child"] != ""].drop_duplicates()

        self.hierarchy_df = df

        # reconcilers config is already a list in your config
        self.reconcilers_cf = config["reconcilers"]

        self.RECONCILER_REGISTRY = {
            "BottomUp": BottomUp,
            "TopDown": TopDown,
            "MiddleOut": MiddleOut,
        }
        self._build_hierarchy_paths()
        self.levels=sorted(list(self.tags.keys()))
        self.spec = [self.levels[:i] for i in range(1, len(self.levels) + 1)]
        


    def _build_hierarchy_paths(self):
        """
        Builds:
        - self.children_of, self.parent_of
        - self.root (first root), self.roots
        - self.nodes (all nodes reachable from root, ordered with bottom last)
        - self.bottom_ids (leaf series)
        - self.S_df (DataFrame with 'unique_id' + bottom columns)
        - self.tags (dict for MiddleOut: level_0, level_1, ...)
        """
    

        children_of = defaultdict(set)
        parent_of = {}

        for p, c in self.hierarchy_df[["parent", "child"]].itertuples(index=False):
            p = str(p)
            c = str(c)

            children_of[p].add(c)

            # enforce single-parent hierarchy
            if c in parent_of and parent_of[c] != p:
                raise ValueError(f"Child '{c}' has multiple parents: {parent_of[c]} and {p}")
            parent_of[c] = p

        # roots: edges with blank parent => "" -> root
        roots = sorted(children_of[""])
        if not roots:
            # fallback: nodes that never appear as a child
            all_nodes = set(self.hierarchy_df["parent"]) | set(self.hierarchy_df["child"])
            all_nodes.discard("")
            roots = sorted([n for n in all_nodes if n not in parent_of])

        if not roots:
            raise ValueError("No root found. Ensure CSV has a blank parent for the root edge.")

        self.children_of = children_of
        self.parent_of = parent_of
        self.roots = roots
        self.root = roots[0]  # pick first root

        # leaves (bottom series): appear as child but never as parent
        all_parents = set(self.hierarchy_df["parent"])
        all_parents.discard("")
        all_children = set(self.hierarchy_df["child"])
        bottom_ids = sorted(all_children - all_parents)
        self.bottom_ids = bottom_ids

        # BFS order nodes from root
        nodes = []
        q = deque([self.root])
        seen = set()

        while q:
            n = q.popleft()
            if n in seen or n == "":
                continue
            seen.add(n)
            nodes.append(n)
            for kid in sorted(children_of.get(n, [])):
                q.append(kid)

        # Make sure all leaves are included even if graph is weird
        for b in bottom_ids:
            if b not in seen:
                nodes.append(b)

        # IMPORTANT: reorder nodes so bottom series are LAST (required by hierarchicalforecast)
        non_bottom = [n for n in nodes if n not in bottom_ids]
        nodes_ordered = non_bottom + bottom_ids

        self.nodes = nodes_ordered

        # depth for tags (level groups)
        depth = {self.root: 0}
        q = deque([self.root])
        while q:
            cur = q.popleft()
            for kid in children_of.get(cur, []):
                if kid not in depth:
                    depth[kid] = depth[cur] + 1
                    q.append(kid)

        # Store depth for later use
        self.depth = depth

        tags = {}
        for n in nodes_ordered:
            d = depth.get(n)
            if d is None:
                continue
            tags.setdefault(f"level_{d}", []).append(n)

        # convert to numpy arrays (what hf expects)
        self.tags = {k: np.array(sorted(v), dtype=object) for k, v in tags.items()}

        # helper: descendant leaves of a node
        def descendant_leaves(node: str) -> set:
            stack = [node]
            out = set()
            while stack:
                cur = stack.pop()
                kids = children_of.get(cur, set())
                if not kids:
                    out.add(cur)
                else:
                    stack.extend(list(kids))
            return out

        # Build S (rows = all nodes ordered, cols = bottom series)
        S = pd.DataFrame(0.0, index=nodes_ordered, columns=bottom_ids)

        # Fill non-bottom rows with descendant structure
        for n in non_bottom:
            desc = descendant_leaves(n)
            cols = [b for b in bottom_ids if b in desc]
            if cols:
                S.loc[n, cols] = 1.0

        # FORCE bottom block = identity (required)
        for b in bottom_ids:
            S.loc[b, :] = 0.0
            S.loc[b, b] = 1.0
        self.S_df = S.reset_index(names="unique_id")

        # Build paths_df: map each bottom_id to its ancestors at each level
        # This enables ensure_hier_cols to add hierarchy columns to bottom-level data
        paths_rows = []
        level_names = sorted(self.tags.keys())  # e.g. ['level_0', 'level_1', 'level_2']
        
        for bottom_id in bottom_ids:
            # Trace path from bottom to root
            path_dict = {"unique_id": bottom_id}
            current = bottom_id
            
            while current in parent_of:
                current = parent_of[current]
                if current and current != "":
                    # Find which level this node is at
                    node_depth = depth.get(current)
                    if node_depth is not None:
                        path_dict[f"level_{node_depth}"] = current
            
            # Fill in any missing levels with the deepest available ancestor
            for i, level_name in enumerate(level_names):
                if level_name not in path_dict:
                    # Find deepest ancestor we have below this level
                    for j in range(i - 1, -1, -1):
                        if f"level_{j}" in path_dict and path_dict[f"level_{j}"] is not None:
                            path_dict[level_name] = path_dict[f"level_{j}"]
                            break
                    # If still not found, it's a gap in the hierarchy
                    if level_name not in path_dict:
                        path_dict[level_name] = None
            
            paths_rows.append(path_dict)
        
        self.paths_df = pd.DataFrame(paths_rows)

    def create_paths_df_for_ids(self, unique_ids):
        """
        Create paths_df for any set of unique_ids, handling orphans gracefully.
        Orphans (IDs not in hierarchy) get placed at the deepest level.
        """
        paths_rows = []
        level_names = sorted(self.tags.keys())
        max_level = len(level_names) - 1
        
        for uid in unique_ids:
            path_dict = {"unique_id": uid}
            
            # Check if this ID is in the hierarchy
            if uid in self.parent_of or any(uid in children for children in self.children_of.values()):
                # ID is in hierarchy - trace path to root
                current = uid
                while current in self.parent_of:
                    current = self.parent_of[current]
                    if current and current != "":
                        node_depth = self.depth.get(current)
                        if node_depth is not None:
                            path_dict[f"level_{node_depth}"] = current
            else:
                # Orphan ID - put it at deepest level
                path_dict[f"level_{max_level}"] = uid
            
            # Fill in missing levels with the deepest available ancestor
            for i, level_name in enumerate(level_names):
                if level_name not in path_dict:
                    # Find deepest ancestor we have below this level
                    for j in range(i - 1, -1, -1):
                        if f"level_{j}" in path_dict and path_dict[f"level_{j}"] is not None:
                            path_dict[level_name] = path_dict[f"level_{j}"]
                            break
                    # If still not found, it's a gap in the hierarchy
                    if level_name not in path_dict:
                        path_dict[level_name] = None
            
            paths_rows.append(path_dict)
        
        result_df = pd.DataFrame(paths_rows)
        return result_df
    
    def _get_node_depth(self, node):
        """Helper to get depth of a node."""
        return self.depth.get(node)

    def build_reconcilers(self):
        reconcilers = []
        for i, item in enumerate(self.reconcilers_cf):
            if "type" not in item:
                raise ValueError(f"reconcilers[{i}] missing 'type'")

            r_type = item["type"]
            params = item.get("params", {}) or {}

            if r_type not in self.RECONCILER_REGISTRY:
                raise ValueError(
                    f"Unknown reconciler type '{r_type}'. "
                    f"Allowed: {list(self.RECONCILER_REGISTRY.keys())}"
                )

            cls = self.RECONCILER_REGISTRY[r_type]
            reconcilers.append(cls(**params))

        return reconcilers