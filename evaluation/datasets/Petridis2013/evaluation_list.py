import glob
import os

_evaluation_list = glob.glob(os.path.dirname(__file__)+"/gt/*.json")

evaluation_list = [os.path.splitext(os.path.basename(file_name))[0] for file_name in _evaluation_list]