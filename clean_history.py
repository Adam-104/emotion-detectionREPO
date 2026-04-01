import json
import uuid

FILE = "history.json"

def clean_history():
    with open(FILE, "r") as f:
        data = json.load(f)

    new_data = []

    for item in data:

        # 🔹 FIX TYPE
        if "type" in item:
            t = item["type"].lower()
        elif "input_type" in item:
            t = item["input_type"].lower()
        else:
            t = "unknown"

        # 🔹 FIX IMAGE PATH
        img = item.get("image", "")
        if img:
            img = img.replace("\\", "/")

            # ensure proper path
            if not img.startswith("/"):
                img = "/" + img

        # 🔹 CREATE CLEAN ENTRY
        new_item = {
            "id": str(uuid.uuid4()),
            "time": item.get("time", ""),
            "type": t,
            "image": img,
            "emotion": item.get("emotion", "Unknown"),
            "age": item.get("age", "N/A"),
            "gender": item.get("gender", "N/A"),
            "suggestion": item.get("suggestion", "")
        }

        new_data.append(new_item)

    # 🔹 SAVE CLEAN FILE
    with open(FILE, "w") as f:
        json.dump(new_data, f, indent=4)

    print("✅ History cleaned successfully!")


if __name__ == "__main__":
    clean_history()