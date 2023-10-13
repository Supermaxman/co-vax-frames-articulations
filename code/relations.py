import argparse
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import networkx as nx

from utilities import (
    read_jsonl,
    write_jsonl,
    format_prompt,
    extract_relations,
    rel_order,
    format_reasoning,
    extract_problems,
    count_problems,
    reduce_paraphrases,
    merge_relations,
    clean_reasoning,
)
from collections import defaultdict

from api import OpenAIAPI


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
        "--prompt_path",
        type=str,
        default="/shared/aifiles/disk1/media/artifacts/cot/co-vax-frames-articulations/annotations",
    )
    arg_parser.add_argument(
        "--art_path", type=str, default="/shared/aifiles/disk1/media/artifacts"
    )
    arg_parser.add_argument("--temperature", type=float, default=0)
    arg_parser.add_argument("--max_tokens", type=int, default=512)
    arg_parser.add_argument("--top_k", type=int, default=10)

    args = arg_parser.parse_args()
    data_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "articulations"
    )
    artifacts_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "relations"
    )
    os.makedirs(artifacts_path, exist_ok=True)

    frames = read_jsonl(
        os.path.join(data_path, "predictions", "articulations-unique.jsonl")
    )

    prompt_messages = read_jsonl(
        os.path.join(
            args.prompt_path, f"relations-{args.split}-{args.method}-prompt.jsonl"
        )
    )

    if args.api == "openai":
        cache_path = os.path.join(artifacts_path, "openai-cache")
        os.makedirs(cache_path, exist_ok=True)
        api = OpenAIAPI(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            delay_seconds=6,
            api_key=args.api_key,
            cache_path=cache_path,
        )
    else:
        raise ValueError(f"Unknown api: {args.api}")

    if args.similarity == "sbert":
        embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown similarity: {args.similarity}")

    a_embs = embed.encode([f["text"] for f in frames], show_progress_bar=True)
    fdists = np.sum((a_embs[:, None] - a_embs[None, :]) ** 2, axis=-1)
    current_mask = np.zeros(shape=[len(frames)], dtype=np.float32)
    current_mask[0] = 1.0
    all_relations = []
    current_index = 1
    responses = []
    with tqdm(total=len(frames)) as pbar:
        pbar.update(current_index)
        while current_index < len(frames):
            ex = frames[current_index]
            ex_dists = fdists[current_index] + (1.0 - current_mask) * 1e6
            f_sorted = np.argsort(ex_dists)[: args.top_k]
            lines = ["Similar known framings:"]
            f_map = {}
            i = 1
            for f_idx in f_sorted:
                if ex_dists[f_idx] > 1e5:
                    continue
                f_text = frames[f_idx]["text"]
                f_text = format_prompt(f_text)
                lines.append(f"{i}: {f_text}")
                f_map[i] = f_idx
                i += 1
            lines.append("New framing:")
            text = format_prompt(ex["text"])
            f_map[i] = current_index
            lines.append(f"{i}: {text}")
            line = "\n".join(lines)
            message = api.build_message(format_prompt(line))
            messages = prompt_messages + [message]
            response = api.send(messages)
            responses.append(response)
            relations = extract_relations(response, f_map)
            if len(relations) == 0:
                # add frame to active frames if no relation
                current_mask[current_index] = 1.0
            all_relations.extend(relations)
            for rel in sorted(relations, key=lambda x: rel_order(x)):
                if rel["type"] == "paraphrases":
                    # only keep shorter, by default we keep the one already in play
                    if len(frames[rel["x"]]["text"]) < len(frames[rel["y"]]["text"]):
                        current_mask[rel["x"]] = 1.0
                        current_mask[rel["y"]] = 0.0
                    break
                elif rel["type"] == "specializes":
                    # keep both specific and general
                    # keep both, so add new one
                    current_mask[current_index] = 1.0
                    break
                elif rel["type"] == "contradicts":
                    # keep both, so add new one
                    current_mask[current_index] = 1.0
                    break
                else:
                    print(f'Unknown relation type: {rel["type"]}')
            current_index += 1

    pred_path = os.path.join(artifacts_path, "predictions")
    os.makedirs(pred_path, exist_ok=True)

    cleaned_relations = []
    for rel in all_relations:
        cleaned_relations.append(
            {
                "type": rel["type"],
                "x": int(rel["x"]),
                "y": int(rel["y"]),
                "reasoning": format_reasoning(rel["reasoning"]),
            }
        )
    rc = defaultdict(int)
    for rel in cleaned_relations:
        rc[rel["type"]] += 1
    for k, v in sorted(rc.items(), key=lambda x: x[1], reverse=True):
        print(k, v)

    write_jsonl(cleaned_relations, os.path.join(pred_path, "relations.jsonl"))
    write_jsonl(responses, os.path.join(pred_path, "responses.jsonl"))
