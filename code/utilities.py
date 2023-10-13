import re
from textwrap import wrap
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
