import json

to_jsonify = {"all_prompts":[]}
with open('./bias_data.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('"')
        idx = line.index('text')
        to_jsonify['all_prompts'].append({'prompt':line[idx + 2]})


json_str = json.dumps(to_jsonify, indent = 4)

with open('bias_data.json', 'w') as j:
    j.write(json_str)

