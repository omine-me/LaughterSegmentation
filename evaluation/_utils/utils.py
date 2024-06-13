def concat_close(laughters, gap_threshold):
    # concat laughters which are close to each other
    laughters_concat = []
    for laughter in laughters.values():
        if len(laughters_concat) == 0:
            laughters_concat.append(laughter.copy())
            continue
        if abs(laughters_concat[-1]["end_sec"] - laughter["start_sec"]) < gap_threshold:
            laughters_concat[-1]["end_sec"] = laughter["end_sec"]
        else:
            laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}

def remove_short(laughters, min_length):
    # remove short laughters
    laughters_concat = []
    for laughter in laughters.values():
        if laughter["end_sec"] - laughter["start_sec"] < min_length:
            continue
        laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}

def remove_inappropriate(laughters):
    # remove inappropriate laughters
    laughters_concat = []
    for laughter in laughters.values():
        if "not_a_laugh" in laughter and laughter["not_a_laugh"]:
            continue
        if laughter["end_sec"] - laughter["start_sec"] <= 0.0:
            continue
        laughters_concat.append(laughter.copy())
    # to dict
    return {str(i): laughter for i, laughter in enumerate(laughters_concat)}
