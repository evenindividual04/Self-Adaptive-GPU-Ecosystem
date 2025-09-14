# json_to_jsonl.py
import argparse, json, sys

def main():
    p = argparse.ArgumentParser(description="Convert JSON array -> JSONL")
    p.add_argument("input_json", help="Path to input JSON (array of objects) or JSONL")
    p.add_argument("output_jsonl", help="Path to write JSONL")
    args = p.parse_args()

    # If input is already JSONL, just copy lines
    if args.input_json.lower().endswith(".jsonl"):
        with open(args.input_json) as r, open(args.output_jsonl, "w") as w:
            for line in r:
                w.write(line if line.endswith("\n") else line + "\n")
        return

    # Otherwise treat as JSON array and convert
    data = json.load(open(args.input_json))  # expects a list[dict]
    if not isinstance(data, list):
        print("Input JSON must be an array of objects", file=sys.stderr); sys.exit(1)
    with open(args.output_jsonl, "w") as w:
        for obj in data:
            w.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    main()
