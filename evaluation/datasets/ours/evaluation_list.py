import glob
import os

_evaluation_list = glob.glob(os.path.dirname(__file__)+"/gt/laugh/*.json")+\
                    glob.glob(os.path.dirname(__file__)+"/gt/non_laugh/*.json")[:201]
assert len(_evaluation_list) == 201 + 201
evaluation_list = [os.path.splitext(os.path.basename(file_name))[0] for file_name in _evaluation_list]