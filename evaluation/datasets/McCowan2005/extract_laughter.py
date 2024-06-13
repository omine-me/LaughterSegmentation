import glob
import json
import xml.etree.ElementTree as ET
from evaluation_list import evaluation_list

transcript_dir = r"./original_anotation_data/words"
output_dir = r"./gt"

all_participants_laughters = []
min_laugh_len = 0.3

def merge_events(event_lists):
    merged_events = {}
    merged_event_idx = 0
    for event_list in event_lists:
        for event in event_list.values():
            if not merged_events:
                # If merged_events is empty, add the first event
                merged_events[str(merged_event_idx)] = event.copy()
                merged_event_idx += 1
            else:
                merged = False
                for merged_event in merged_events.values():
                    if event["start_sec"] <= merged_event["end_sec"] and event["end_sec"] >= merged_event["start_sec"]:
                        # Events overlap, merge them
                        merged_event["start_sec"] = min(event["start_sec"], merged_event["start_sec"])
                        merged_event["end_sec"] = max(event["end_sec"], merged_event["end_sec"])
                        merged = True
                        # break
                if not merged:
                    # If the event does not overlap with any merged event, add it to merged_events
                    merged_events[str(merged_event_idx)] = event.copy()
                    merged_event_idx += 1
    if len(event_lists) > 1:
        merged_events = merge_events([merged_events])
    merged_events = sorted(merged_events.values(), key=lambda x: x["start_sec"])
    merged_events = {str(idx): val for idx, val in enumerate(merged_events)}
    return merged_events

for test_data in evaluation_list:
    participants_transcripts = glob.glob(transcript_dir + f"/{test_data}*.xml")
    if not participants_transcripts:
        continue

    laughters = []
    for transcript_path in participants_transcripts:
        laughter = {}
        laughter_idx = 0
        for event in ET.parse(transcript_path).getroot():
            if "type" in event.attrib and event.attrib["type"] == "laugh":
                # skip unannotated laughs
                if "starttime" not in event.attrib or "endtime" not in event.attrib:
                    continue
                # skip short laughs
                if float(event.attrib["endtime"]) - float(event.attrib["starttime"]) < min_laugh_len:
                    continue

                laughter[str(laughter_idx)] = {
                    "start_sec": float(event.attrib["starttime"]),
                    "end_sec": float(event.attrib["endtime"]),
                }
                laughter_idx += 1
        laughters.append(laughter)
    
    merged_laughter = merge_events(laughters)

    with open(output_dir + f"/{test_data}.Mix-Headset.json", "w", encoding="utf-8") as f:
        json.dump(merged_laughter, f)