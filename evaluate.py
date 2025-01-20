import argparse
from render2loc.lib import  eval
import json
import os
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(os.getcwd())
    parser.add_argument('--config_file',default='./config/config_evaluate.json', type=str, help='configuration file')
    args = parser.parse_args()
     
    with open(args.config_file) as fp:
        config = json.load(fp)
    
    eval.main(config["evaluate"])
    
    # iterative_relocalization.main(config["render2loc"])

    # # render by Render2loc's results
    # config["render2loc"].update({
    #     "input_pose": config["render2loc"]["results"],
    #     "depth_path"  : config["render2loc"]["final_depth_path"]
    #     })
    # render_pyrender.main(config["render2loc"])
