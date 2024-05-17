# python test_LLaVA.py --model-path /mnt/data/plm/llava-v1.5-13b --load-8bit --device cuda:5 --file-path ./data/DeepSemantics_Questions.json --save-path ./result/DeepSemantics_Prompt_1_LLaVA-13B.json --prompt Prompt_1

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
from io import BytesIO
import os
from PIL import Image
import os
from transformers import TextStreamer
from tqdm import tqdm
import urllib.request


def main(args):
    
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    # conv mode 
    if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    args.conv_mode = conv_mode

    # load tokenizer, model, and image_processor 
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

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
        image_tensor = process_images([image], image_processor, args)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Prompt
        inp = triple[args.prompt].format(triple['Options']['A'],triple['Options']['B'],triple['Options']['C'],triple['Options']['D'])
        
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        outputs = ''
        N = 0
        while ("A\n" not in outputs) and ("A," not in outputs) and ("A<" not in outputs) and ("A." not in outputs) and ("A)" not in outputs) and ("B\n" not in outputs) and ("B," not in outputs) and ("B<" not in outputs) and ("B." not in outputs) and ("B)" not in outputs) and ("C\n" not in outputs) and ("C," not in outputs) and ("C<" not in outputs) and ("C." not in outputs) and ("C)" not in outputs) and ("D\n" not in outputs) and ("D," not in outputs) and ("D<" not in outputs) and ("D." not in outputs) and ("D)" not in outputs):
            N += 1
            if N > 5:
                break
            
            conv = conv_templates[args.conv_mode].copy()
            roles = conv.roles
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            outputs = outputs.strip()

        # print(outputs)
        output = {"Triple":triple, 'Model_Path':args.model_path, "Output":outputs}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')       

    save_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_LLaVA-13B.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
