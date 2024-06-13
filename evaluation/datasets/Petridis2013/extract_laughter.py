import csv
import json
import os

output_dir = r"./gt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load csv
with open("./original_anotation_data/annotations.csv", newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    laughter_list = list(csvreader)
    # sort with first column. if the first column is the same, sort with second column
    laughter_list = sorted(laughter_list, key=lambda x: (x[0], float(x[1])))
    
    laughters = {}
    laughter_idx = 0

    for idx, row in enumerate(laughter_list):
        laughters[str(laughter_idx)] = {
                        "start_sec": float(row[1]),
                        "end_sec": float(row[2]),
                    }
        laughter_idx += 1

        # if the next row is not the same participant, save the laughters
        if idx == len(laughter_list)-1 or row[0] != laughter_list[idx+1][0]:
            laughters = sorted(laughters.values(), key=lambda x: x["start_sec"])
            laughters = {str(idx): val for idx, val in enumerate(laughters)}
            with open(output_dir + f"/{row[0]}_cam.json", "w", encoding="utf-8") as f:
                json.dump(laughters, f)
            laughters = {}
            laughter_idx = 0
