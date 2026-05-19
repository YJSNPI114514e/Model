import json
import os

log_path = r"C:\Users\gekik\.gemini\antigravity\brain\d81832e5-56e3-4a09-94c1-71809a15af3a\.system_generated\logs\overview.txt"
output_path = r"c:\Users\gekik\Downloads\Model-main\Model\scratch\extracted_changes.txt"

with open(log_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out:
    for line in f:
        try:
            data = json.loads(line)
        except Exception:
            continue
        if 'tool_calls' in data:
            for tc in data['tool_calls']:
                if tc['name'] in ('replace_file_content', 'write_to_file'):
                    args = tc['args']
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    out.write("=" * 60 + "\n")
                    out.write(f"Tool: {tc['name']}\n")
                    out.write(f"File: {args.get('TargetFile')}\n")
                    out.write(f"StartLine: {args.get('StartLine')}\n")
                    out.write(f"EndLine: {args.get('EndLine')}\n")
                    out.write(f"TargetContent:\n{args.get('TargetContent')}\n")
                    out.write("-" * 40 + "\n")
                    out.write(f"ReplacementContent:\n{args.get('ReplacementContent')}\n")
                    if args.get('CodeContent'):
                        out.write(f"CodeContent (first 500 chars):\n{args.get('CodeContent')[:500]}\n")
                    out.write("\n")

print("Extraction completed successfully.")
