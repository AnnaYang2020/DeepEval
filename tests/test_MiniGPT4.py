# python test_MiniGPT4.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2,StoppingCriteriaSub

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
import json
from tqdm import tqdm
import urllib.request


def main(args):
    # Model Initialization
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    file_ = open(args.file_path, 'r')
    entities = file_.read().strip().split("\n")
    file_.close()

    save_file = open(args.save_path, 'a')


    for entity in tqdm(entities):

        # load as dict
        triple = json.loads(entity.strip())

        # Image
        image_url = triple['Url']
        urllib.request.urlretrieve(image_url, triple['Image_ID'])
        image =  Image.open(triple['Image_ID']).convert('RGB')
    
        llm_message = ""
        while ("A\n" not in llm_message) and ("A," not in llm_message) and ("A<" not in llm_message) and ("A." not in llm_message) and ("A)" not in llm_message) and ("B\n" not in llm_message) and ("B," not in llm_message) and ("B<" not in llm_message) and ("B." not in llm_message) and ("B)" not in llm_message) and ("C\n" not in llm_message) and ("C," not in llm_message) and ("C<" not in llm_message) and ("C." not in llm_message) and ("C)" not in llm_message) and ("D\n" not in llm_message) and ("D," not in llm_message) and ("D<" not in llm_message) and ("D." not in llm_message) and ("D)" not in llm_message):
            chat_state = CONV_VISION.copy() # chat_state.messages == []
            img_list = []

            llm_message = chat.upload_img(image, chat_state, img_list)
            chat.encode_img(img_list)

            # ask
            user_message = triple[args.prompt].format(triple['Options']['A'],triple['Options']['B'],triple['Options']['C'],triple['Options']['D'])
            chat.ask(user_message, chat_state)

            # answer
            num_beams = args.num_beams
            temperature = args.temperature
            llm_message = chat.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=num_beams,
                                        temperature=temperature,
                                        max_new_tokens=args.max_new_tokens,
                                        max_length=args.max_length)[0]
            

        output = {"Triple":triple, 'Model_Path':args.cfg_path, "Output":llm_message}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')
        

    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.") # eval_configs/minigpt4_llama2_eval.yaml
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_MiniGPT4-13B.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=2000, help='max length of the total sequence')
    parser.add_argument("--num_beams", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0, help='temperature for sampling')
    args = parser.parse_args()

    main(args)

    

