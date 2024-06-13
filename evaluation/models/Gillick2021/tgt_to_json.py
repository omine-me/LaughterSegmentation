import glob
import os
import json
import tgt

def main(referenced_paper):
    tgt_dir = rf"./{referenced_paper}/textgrid"

    tgt_files = glob.glob(tgt_dir + "/*.TextGrid")

    for tgt_file in tgt_files:
        laughter = {}
        tgt_obj = tgt.io.read_textgrid(tgt_file)
        tgt_laughs = tgt_obj.tiers[0]
        for idx, tgt_laugh in enumerate(tgt_laughs):
            laughter[str(idx)] = {"start_sec": tgt_laugh.start_time,
                                "end_sec": tgt_laugh.end_time,
                                }
        
        out_path = f"./{referenced_paper}/" + os.path.splitext(os.path.basename(tgt_file))[0].removesuffix("_laughter") + ".json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(laughter, f)

if __name__ == "__main__":
    main("Petridis2013")