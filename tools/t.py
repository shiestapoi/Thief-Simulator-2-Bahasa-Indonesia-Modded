import json

with open("LocalizationFile_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("LocalizationFile_merged.json", "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":" ), ensure_ascii=False)
