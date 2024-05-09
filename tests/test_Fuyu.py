from transformers import FuyuProcessor, FuyuForCausalLM
import torch
from PIL import Image

import os
import json
import argparse
import glob
from tqdm import tqdm
import urllib.request


def main(args):
    
    print(f"Start loading fuyu\n")
    #model_path = "/home/lz/model/fuyu-8b"
    
    with torch.no_grad():
        processor = FuyuProcessor.from_pretrained(args.model_path)
        model = FuyuForCausalLM.from_pretrained(args.model_path).half().to(args.device)
        # model = torch.nn.DataParallel(model,device_ids=[0,1],output_device=0)

        # load file
        file_ = open(arg.file_path, 'r')
        entities = file_.read().strip().split("\n")
        file_.close()

        # save file
        save_file = open(arg.save_path, 'a')

        for entity in tqdm(entities):
            # load as dict
            triple = json.loads(entity.strip())

            # Prompt
            question = triple[arg.prompt].format(triple['options']['A'],triple['options']['B'],triple['options']['C'],triple['options']['D'])

            # Image
            image_url = triple['Url']
            urllib.request.urlretrieve(image_url, triple['Image_ID'])
            image = Image.open(triple['Image_ID']).convert('RGB')
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(args.device)

            # Output
            generation_output = model.module.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generation_text = processor.batch_decode(generation_output[:, -args.max_new_tokens:], skip_special_tokens=True)
            ans = generation_text[0]

            output = {"Triple":triple, 'Model_Path':args.model_path, "Output":ans}
            sen_dict = json.dumps(output)
            save_file.write(sen_dict + '\n')       

        save_file.close()
        print("end")


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_Fuyu.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=7)
    args = parser.parse_args()
    main(args)
