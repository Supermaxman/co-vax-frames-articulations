import argparse
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import networkx as nx
import ujson as json

from utilities import (
    read_jsonl,
    write_jsonl,
    extract_problems,
    count_problems,
    reduce_paraphrases,
    merge_relations,
    clean_reasoning,
)
from annotate import annotate_frames, annotate_relations


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str)
    arg_parser.add_argument("--similarity", type=str, default="sbert")
    arg_parser.add_argument("--method", type=str)
    arg_parser.add_argument("--api", type=str)
    arg_parser.add_argument("--api_key", type=str, default=None)
    arg_parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    arg_parser.add_argument(
        "--data_path",
        type=str,
        default="/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/co-vax-frames.json",
    )
    arg_parser.add_argument(
        "--art_path", type=str, default="/shared/aifiles/disk1/media/artifacts"
    )

    args = arg_parser.parse_args()
    artifacts_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "relevance"
    )
    os.makedirs(artifacts_path, exist_ok=True)

    art_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "articulations"
    )
    frames = read_jsonl(
        os.path.join(art_path, "predictions", "articulations-unique.jsonl")
    )

    rel_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "relations"
    )

    cleaned_relations = read_jsonl(
        os.path.join(rel_path, "predictions", "relations.jsonl")
    )

    pred_path = os.path.join(artifacts_path, "predictions")
    os.makedirs(pred_path, exist_ok=True)

    if args.similarity == "sbert":
        embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown similarity: {args.similarity}")

    count_problems(frames)

    print(f"Found {len(frames)} frames before paraphrase reduction")
    print(f"Found {len(cleaned_relations)} relations before paraphrase reduction")
    reduced_frames, reduced_relations, kept_nodes, reduced_count = reduce_paraphrases(
        frames, cleaned_relations
    )
    print(f"Found {len(reduced_frames)} frames after paraphrase reduction")
    print(f"Found {len(reduced_relations)} relations after paraphrase reduction")

    with open(os.path.join(pred_path, "reduced-frames.json"), "w") as f:
        json.dump(reduced_frames, f)
    with open(os.path.join(pred_path, "reduced-count.json"), "w") as f:
        json.dump(reduced_count, f)
    write_jsonl(reduced_relations, os.path.join(pred_path, "reduced-relations.jsonl"))

    count_problems(reduced_frames.values())

    merged_frames, merged_relations, merged_nodes, merged_count = merge_relations(
        frames, reduced_relations, kept_nodes, reduced_count
    )
    print(f"Found {len(merged_frames)} frames after merge reduction")
    print(f"Found {len(merged_relations)} relations after merge reduction")

    count_problems(merged_frames.values())
    with open(os.path.join(pred_path, "merged-frames.json"), "w") as f:
        json.dump(merged_frames, f)
    with open(os.path.join(pred_path, "merged-count.json"), "w") as f:
        json.dump(merged_count, f)
    write_jsonl(merged_relations, os.path.join(pred_path, "merged-relations.jsonl"))

    new_frames = []
    for f_idx, f in merged_frames.items():
        f = f.copy()
        f["count"] = merged_count[f_idx]
        f["f_id"] = f"F{f_idx}"
        f["problems"] = extract_problems(f)
        f["reasoning"] = clean_reasoning(f["reasoning"])
        new_frames.append(f)

    new_relations = []
    for rel in merged_relations:
        new_rel = {
            "x": "F" + str(rel["x"]),
            "y": "F" + str(rel["y"]),
            "type": rel["type"],
            "reasoning": rel["reasoning"],
        }
        new_relations.append(new_rel)

    write_jsonl(new_frames, os.path.join(pred_path, "relevant-frames.jsonl"))
    write_jsonl(new_relations, os.path.join(pred_path, "relevant-relations.jsonl"))

    with open(args.data_path) as f:
        r_frames = json.load(f)
        ref_frames = []
        for f_id, f in r_frames.items():
            f["f_id"] = f_id
            ref_frames.append(f)
    ann_frames_path = os.path.join(
        pred_path, f"{args.model}-{args.method}-{args.split}-frames.xlsx"
    )
    annotate_frames(
        merged_frames,
        merged_relations,
        merged_count,
        ann_frames_path,
        embed,
        ref_frames,
    )
    ann_rel_path = os.path.join(
        pred_path, f"{args.model}-{args.method}-{args.split}-rels.xlsx"
    )
    annotate_relations(merged_frames, merged_relations, merged_count, ann_rel_path)
    print(ann_rel_path)
