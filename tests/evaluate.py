import json
import copy
import re
from tqdm import tqdm
import argparse

def main(args):
    pattern = r'(?<![a-zA-Z])[ABCD](?![a-zA-Z])'

    path = args.result_path
    file = open(path, 'r')
    entities = file.read().strip().split("\n")
    file.close()

    flag = 0
    count = 0
    count_all = 0
    for entity in tqdm(entities):
        count_all+=1
        
        entity = json.loads(entity.strip())
        answer_true = entity["Triple"]["Answer"]
        answer = entity["Output"].replace("\u0004","")
        answer = answer.strip()
        options = entity["Triple"]["Options"]

        ans = None
        tmp = None
        if options["A"].strip()[:-1].lower() in answer.lower():
            ans = "A"
        elif options["B"].strip()[:-1].lower() in answer.lower():
            ans = "B"
        elif options["C"].strip()[:-1].lower() in answer.lower():
            ans = "C"
        elif options["D"].strip()[:-1].lower() in answer.lower():
            ans = "D"
        if ans==None:
            matches = re.findall(pattern, answer)
            if matches:
                first_letter = matches[0]
                ans = first_letter
            else:
                ans = "Unknown Option"
                flag+=1

        if ans == answer_true:
            count+=1

    result = count/count_all*100
    print(path)
    print("Accuracy: %.2f"%result,"%")    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, default="./result/DeepSemantics_Prompt_1_LLaVA-13B.json")
    args = parser.parse_args()
    main(args)
