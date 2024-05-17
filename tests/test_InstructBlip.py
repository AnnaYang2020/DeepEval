import argparse
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import os
import json
import glob
from tqdm import tqdm
import urllib.request

def main(args):
    print(f"Start loading instructblip\n")
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).half().to(args.device)
    processor = InstructBlipProcessor.from_pretrained(args.model_path)

    # load file
    file_ = open(args.file_path, 'r')
    entities = file_.read().strip().split("\n")
    file_.close()

    # save file
    save_file = open(args.save_path, 'a')

    for entity in tqdm(entities):
        # load as dict
        triple = json.loads(entity.strip())

        # Image
        image_url = triple['Url']
        urllib.request.urlretrieve(image_url, triple['Image_ID'])
        image =  Image.open(triple['Image_ID']).convert('RGB')

        # prompt
        question = triple[args.prompt].format(triple['Options']['A'],triple['Options']['B'],triple['Options']['C'],triple['Options']['D'])

        # generate
        inputs = processor(images=image, text=prompt, return_tensors="pt", max_length=512, truncation=True).to(args.device)
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=args.num_beams,
                max_length=args.max_length,
                min_length=1,
                top_p=args.top_p,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=args.temperature,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        output = {"Triple":triple, 'Model_Path':args.model_path, "Output":generated_text}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')       

    save_file.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_InstructBlip-13B.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
