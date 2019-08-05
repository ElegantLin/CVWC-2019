import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert to submission version")
    parser.add_argument('-o', '--original', default='result.pickle.json')
    parser.add_argument('-t', '--target', default='result.json')
    args = parser.parse_args()

    with open(args.original, 'r') as f:
        local_dict = json.load(f)

    for i in local_dict:
        i['image_id'] = int(i['image_id'])

    with open(args.target, "w+") as f:
        json.dump(local_dict, f)
