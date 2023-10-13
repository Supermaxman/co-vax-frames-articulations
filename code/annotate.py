import pandas as pd
import string
from textwrap import wrap
from utilities import order_hierarchy, extract_problems
import numpy as np


def create_excel(data, output_path, columns=None, labels=None):
    labels = labels.copy()
    if columns is None:
        columns = list(data[0].keys())
    column_types = {}
    for label_letter, label_name in zip(string.ascii_uppercase, labels):
        columns.append(label_name)
        labels[label_name]["label_letter"] = label_letter
    # TODO support more than A-Z columns
    for column_letter, column_name in zip(string.ascii_uppercase, columns):
        if column_name == "id" or column_name.endswith("_id"):
            column_types[column_letter] = "id"
        elif column_name in labels:
            labels[column_name]["data_letter"] = column_letter
            column_types[column_letter] = "label"
        else:
            column_types[column_letter] = "text"

    df = pd.DataFrame(data=data, columns=columns)
    df = df.fillna("")
    df = df.set_index(columns[0])
    data_size = len(df)
    l_df = pd.DataFrame({l_name: l["values"] for l_name, l in labels.items()})

    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="data")
    l_df.to_excel(writer, sheet_name="labels", index=False)
    workbook = writer.book
    worksheet = writer.sheets["data"]
    for label_name, label in labels.items():
        data_letter = label["data_letter"]
        label_letter = label["label_letter"]
        label_values = label["values"]
        label_message = label["message"]
        worksheet.data_validation(
            f"{data_letter}2:{data_letter}{data_size+1}",
            {
                "validate": "list",
                "source": f"=labels!${label_letter}$2:${label_letter}${len(label_values)+1}",
                "input_message": label_message,
            },
        )
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    for column_letter, column_type in column_types.items():
        column_range = f"{column_letter}:{column_letter}"
        # first_col, last_col, width, cell_format, options
        if column_type == "id":
            worksheet.set_column(column_range, options={"hidden": True})
        elif column_type == "text":
            worksheet.set_column(column_range, cell_format=cell_format, width=55)
        elif column_type == "label":
            worksheet.set_column(column_range, width=15)

    writer.close()


def annotate_frames(frames, relations, counts, name, ref_frames, embed):
    r_embs = embed.encode([f["text"] for f in ref_frames])
    samples = []
    fs = list(order_hierarchy(frames, relations, counts))
    fs_embs = embed.encode([f["text"] for f_id, f in fs])
    # TODO grab closest known framing by emb distance
    seen_f_ids = set()
    for (f_id, f), f_emb in zip(fs, fs_embs):
        if f_id in seen_f_ids:
            continue
        seen_f_ids.add(f_id)
        fc = counts[f_id]
        problems = extract_problems(f)
        p_str = ", ".join([p.title() for p in problems])
        cf_idx = np.argmin(np.sum((f_emb[None, :] - r_embs) ** 2, axis=-1))
        cf = ref_frames[cf_idx]
        samples.append(
            {
                "f_id": f_id,
                "ref_f_id": cf["f_id"],
                "count": fc,
                "problems": p_str,
                "text": "\n".join(wrap(f["text"], 50)),
                "ref_text": "\n".join(wrap(cf["text"], 50)),
            }
        )
    # Same or specialize or contradict
    #
    print(len(seen_f_ids))
    create_excel(
        samples,
        name,
        labels={
            "Good": {
                "values": ["Yes", "No", "N/A"],
                "message": "Is this discovered frame well-articulated?",
            },
            "Known": {
                "values": ["Yes", "No", "N/A"],
                "message": "Is this discovered frame the same as the known frame?",
            },
            "Better": {
                "values": ["Better", "Same", "Worse"],
                "message": "Is this discovered frame articulated better, worse, or the same as the known frame?",
            },
            "Problems": {
                "values": ["Yes", "No", "N/A"],
                "message": "Are the discovered problems correct?",
            },
        },
    )


def annotate_relations(frames, relations, counts, name):
    samples = []
    rs = sorted(relations, key=lambda x: (x["x"], x["y"]))
    for rel in rs:
        fx = frames[rel["x"]]
        fy = frames[rel["y"]]
        ts = rel["type"].title()
        r_id = f"{rel['type']}-{rel['x']}-{rel['y']}"

        samples.append(
            {
                "r_id": r_id,
                "fx_text": "\n".join(wrap(fx["text"], 50)),
                "ts_text": ts,
                "fy_text": "\n".join(wrap(fy["text"], 50)),
            }
        )
    create_excel(
        samples,
        name,
        labels={
            "Correct": {
                "values": ["Yes", "No"],
                "message": "Is this discovered relation correct?",
            },
        },
    )
