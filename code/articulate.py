import argparse
import os
from tqdm import tqdm

from utilities import read_jsonl, write_jsonl, format_text, extract_frames, build_api
from collections import defaultdict


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str)
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
        "--data_path",
        type=str,
        default="/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/co-vax-frames-test.jsonl",
    )
    arg_parser.add_argument(
        "--art_path", type=str, default="/shared/aifiles/disk1/media/artifacts"
    )
    arg_parser.add_argument("--temperature", type=float, default=0)
    arg_parser.add_argument("--max_tokens", type=int, default=512)

    args = arg_parser.parse_args()

    artifacts_path = os.path.join(
        args.art_path, f"{args.model}-{args.method}-{args.split}", "articulations"
    )
    os.makedirs(artifacts_path, exist_ok=True)

    data = read_jsonl(args.data_path)

    prompt_messages = read_jsonl(
        os.path.join(
            args.prompt_path, f"articulation-{args.split}-{args.method}-prompt.jsonl"
        )
    )

    api = build_api(args, artifacts_path)

    responses = []
    all_articulations = []
    articulated_examples = []
    annotations = []
    for ex in tqdm(data):
        text = format_text(ex["text"])
        message = api.build_message(text)
        messages = prompt_messages + [message]
        response = api.send(messages)
        responses.append(response)
        articulations = extract_frames(response)
        ex["articulations"] = articulations
        articulated_examples.append(ex)
        all_articulations.extend(articulations)
        ann = {"id": ex["id"], "articulations": articulations}
        annotations.append(ann)

    seen = set()
    dup_count = defaultdict(int)
    unique_articulations = []
    for f in all_articulations:
        dup_count[f["text"]] += 1
        if f["text"] in seen:
            continue
        seen.add(f["text"])
        unique_articulations.append(f)
    for frame in unique_articulations:
        frame["count"] = dup_count[frame["text"]]

    pred_path = os.path.join(artifacts_path, "predictions")
    os.makedirs(pred_path, exist_ok=True)

    write_jsonl(all_articulations, os.path.join(pred_path, "articulations-full.jsonl"))
    write_jsonl(
        unique_articulations, os.path.join(pred_path, "articulations-unique.jsonl")
    )
    write_jsonl(
        articulated_examples, os.path.join(pred_path, "articulation-examples.jsonl")
    )
    write_jsonl(annotations, os.path.join(pred_path, "articulation-annotations.jsonl"))
    write_jsonl(responses, os.path.join(pred_path, "responses.jsonl"))

    print(f"Articulated {len(articulated_examples)} examples")
    print(f"Found {len(all_articulations)} frames")
    print(f"Found {len(unique_articulations)} unique frames")
