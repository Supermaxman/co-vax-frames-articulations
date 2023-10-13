import re
from textwrap import wrap
from collections import defaultdict
import networkx as nx

import ujson as json


def read_jsonl(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ex = json.loads(line)
                    examples.append(ex)
                except Exception as e:
                    print(e)
    return examples


def write_jsonl(data, path):
    with open(path, "w") as f:
        for example in data:
            json_data = json.dumps(example)
            f.write(json_data + "\n")


def format_prompt(text):
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def format_text(text):
    text = re.sub(r"http\S+", "", text).strip()
    text = re.sub(r"@[A-Za-z0-9._-]+", "@USER", text).strip()
    text = text.replace("&amp;", "&")
    return format_prompt(text)


def print_messages(messages, usage=None):
    for message in messages:
        print(
            f'{message["role"].title()}'
            + ("" if usage is None else f" ({100*usage:.0f}% usage)")
        )
        for line in message["content"].split("\n"):
            for idx, l in enumerate(wrap(line)):
                print(" " * (2 if idx == 0 else 4) + l)


def extract_frames(message):
    found_frames = []
    text = message["content"]
    reasoning = None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            m_id = line.split(":")[0]
            mf_id, mt = m_id.split(".")
        except:
            return []
        content = line[len(m_id) + 1 :].strip()
        if mt == "a":
            reasoning = content
        if mt == "b":
            found_frames.append({"text": content, "reasoning": reasoning})
    return found_frames


def extract_relations(response, f_map):
    content = response["content"]
    relations = []
    try:
        reasoning = None
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            mt = line.split(":")[0]
            content = line[len(mt) + 1 :].strip()
            if mt == "a":
                reasoning = content
            if mt == "b":
                # Paraphrases(X,Y)
                # Specializes(X,Y)
                # Contradicts(X,Y)
                rt, c = content.split("(")
                rt = rt.lower()
                x, y = c[:-1].split(",")
                relations.append(
                    {
                        "type": rt,
                        "x": f_map[int(x)],
                        "y": f_map[int(y)],
                        "reasoning": reasoning,
                    }
                )
    except Exception as e:
        pass
    return relations


def rel_order(rel):
    if rel["type"] == "paraphrases":
        return -1
    elif rel["type"] == "specializes":
        return 0
    elif rel["type"] == "contradicts":
        return 1
    else:
        return 2


def format_reasoning(r):
    bt_idx = r.find(" between")
    as_idx = r.find(", as")
    if bt_idx != -1 and as_idx != -1:
        r = r[:bt_idx] + r[as_idx:]
    return r.strip()


def extract_problems(frame):
    r = frame["reasoning"].lower()
    pairs = []
    if "confidence" in r:
        pairs.append("confidence")
    if "conspiracy" in r:
        pairs.append("conspiracy")
    if "complacency" in r:
        pairs.append("complacency")
    if "calculation" in r:
        pairs.append("calculation")
    if "collective" in r:
        pairs.append("collective")
    if "compliance" in r:
        pairs.append("compliance")
    if "constraints" in r:
        pairs.append("constraints")
    if len(pairs) == 0:
        pairs.append("other")
    return pairs


def count_problems(frames):
    counts = defaultdict(int)
    all_pairs = set()
    for frame in frames:
        r = frame["reasoning"].lower()
        pairs = []
        if "confidence" in r:
            counts["Confidence"] += 1
            pairs.append("Confidence")
        if "conspiracy" in r:
            counts["Conspiracy"] += 1
            pairs.append("Conspiracy")
        if "complacency" in r:
            counts["Complacency"] += 1
            pairs.append("Complacency")
        if "calculation" in r:
            counts["Calculation"] += 1
            pairs.append("Calculation")
        if "collective" in r:
            counts["Collective"] += 1
            pairs.append("Collective")
        if "compliance" in r:
            counts["Compliance"] += 1
            pairs.append("Compliance")
        if "constraints" in r:
            counts["Constraints"] += 1
            pairs.append("Constraints")
        if len(pairs) == 0:
            pairs.append("Other")
            counts["Other"] += 1
            print(frame["reasoning"])
        all_pairs.add(tuple(pairs))
    for c, count in counts.items():
        print(f"{c}: {count} ({100*count/len(frames):.0f}%)")

    print(all_pairs)


def reduce_paraphrases(frames, relations):
    g = nx.Graph()

    for f_idx, frame in enumerate(frames):
        g.add_node(f_idx)

    for edge in relations:
        if edge["type"] == "paraphrases":
            g.add_edge(edge["x"], edge["y"])

    kept_nodes = set()
    node_map = {}
    reduced_count = defaultdict(int)
    for c in nx.connected_components(g):
        max_node = None
        max_deg = -1
        for n in c:
            d = g.degree[n]
            if d > max_deg:
                max_deg = d
                max_node = n
        kept_nodes.add(max_node)
        for n in c:
            node_map[n] = max_node
            reduced_count[max_node] += frames[n]["count"]

    reduced_relations = []
    for edge in relations:
        if edge["type"] != "paraphrases":
            reduced_relations.append(
                {
                    "type": edge["type"],
                    "x": node_map[edge["x"]],
                    "y": node_map[edge["y"]],
                    "reasoning": edge["reasoning"],
                }
            )

    reduced_frames = {f_idx: f for f_idx, f in enumerate(frames) if f_idx in kept_nodes}
    return reduced_frames, reduced_relations, kept_nodes, reduced_count


def merge_relations(frames, reduced_relations, kept_nodes, reduced_count):
    min_count = 2
    g = nx.DiGraph()
    cg = nx.Graph()

    for f_idx, frame in enumerate(frames):
        if f_idx not in kept_nodes:
            continue
        g.add_node(f_idx)
        cg.add_node(f_idx)

    cr_map = {}
    sr_map = {}
    for edge in reduced_relations:
        if (
            edge["type"] == "specializes"
            and edge["x"] in kept_nodes
            and edge["y"] in kept_nodes
        ):
            g.add_edge(edge["x"], edge["y"])
            sr_map[(edge["x"], edge["y"])] = edge["reasoning"]
        if (
            edge["type"] == "contradicts"
            and edge["x"] in kept_nodes
            and edge["y"] in kept_nodes
        ):
            cg.add_edge(edge["x"], edge["y"])
            cr_map[(edge["x"], edge["y"])] = edge["reasoning"]

    merged_count = reduced_count.copy()
    merged_relations = reduced_relations.copy()
    merged = set()
    deleted_top = 0
    for n in g.nodes():
        count = reduced_count[n]
        in_edges = list(g.in_edges(n))
        out_edges = list(g.out_edges(n))
        c_neighbors = list(cg.neighbors(n))
        if count < min_count:
            merged.add(n)
            if len(out_edges) == 0:
                deleted_top += count
            for _, v in out_edges:
                for u in c_neighbors:
                    merged_relations.append(
                        {
                            "type": "contradicts",
                            "x": u,
                            "y": v,
                            "reasoning": cr_map[(n, u)]
                            if (n, u) in cr_map
                            else cr_map[(u, n)],
                        }
                    )
                merged_count[v] += count
                for u, _ in in_edges:
                    merged_relations.append(
                        {
                            "type": "specializes",
                            "x": u,
                            "y": v,
                            "reasoning": sr_map[(n, v)],
                        }
                    )
    merged_relations = [
        r for r in merged_relations if r["x"] not in merged and r["y"] not in merged
    ]
    merged_frames = {
        f_idx: frame
        for f_idx, frame in enumerate(frames)
        if f_idx in kept_nodes and f_idx not in merged
    }
    return merged_frames, merged_relations, merged, merged_count


def clean_reasoning(r):
    as_idx = r.find(", as ")
    if as_idx != -1:
        as_idx += len(", as ")
        r = r[as_idx:]
    return r.strip().capitalize()


def order_hierarchy(frames, relations, counts, max_depth=None):
    g = nx.DiGraph()
    ig = nx.DiGraph()

    for f_idx, frame in frames.items():
        g.add_node(f_idx)
        ig.add_node(f_idx)

    for edge in relations:
        if edge["type"] == "specializes":
            g.add_edge(edge["x"], edge["y"])
            ig.add_edge(edge["y"], edge["x"])

    roots = set()
    for n, d in g.out_degree():
        if d == 0:
            roots.add(n)

    r_count = {}

    def recursive_count(f_idx):
        if f_idx in r_count:
            return r_count[f_idx]
        frame = frames[f_idx]
        count = counts[f_idx]
        edges = list(ig.out_edges(f_idx))
        for idx, (_, v) in enumerate(edges):
            if v not in r_count:
                r_count[v] = recursive_count(v)
            count += r_count[v]
        return count

    ordered = []

    def recursive_frame_print(f_idx, depth=0, last=False, prefix="", max_depth=None):
        if max_depth is not None and depth >= max_depth:
            return
        frame = frames[f_idx]
        problems = extract_problems(frame)
        p_str = ", ".join([p.title() for p in problems])
        count = recursive_count(f_idx)
        new_prefix = prefix + ("│  " if not last else "   ")
        ordered.append((f_idx, frame))
        edges = list(ig.out_edges(f_idx))
        for idx, (_, v) in enumerate(
            sorted(edges, key=lambda x: recursive_count(x[1]), reverse=True)
        ):
            recursive_frame_print(
                v,
                depth + 1,
                last=idx == len(edges) - 1,
                prefix=new_prefix,
                max_depth=max_depth,
            )

    for idx, f_idx in enumerate(
        sorted(list(roots), key=lambda x: recursive_count(x), reverse=True)
    ):
        recursive_frame_print(f_idx, 0, last=idx == len(roots) - 1, max_depth=max_depth)
    return ordered
