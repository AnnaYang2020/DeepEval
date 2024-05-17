from models.cogvlm_model import CogVLMModel
from utils.language import llama2_tokenizer, llama2_text_processor_inference
from utils.vision import get_image_processor
from utils.chat import chat
from sat.model.mixins import CachedAutoregressiveMixin
import argparse
import torch
import torch.nn as nn
import os
import json
from sat.mpu import get_model_parallel_world_size
from tqdm import tqdm

def main(args):
    # load model
    model, model_args = CogVLMModel.from_pretrained(
        args.model_path,
        args=argparse.Namespace(
            deepspeed=None,
            local_rank=0,
            rank=0,
            world_size=args.world_size,
            model_parallel_size=args.world_size,
            mode='inference',
            skip_init=True,
            fp16=args.fp16,
            bf16=args.bf16,
            use_gpu_initialization=True,
            device=args.device,
        ), overwrite_args={'model_parallel_size': args.world_size} if args.world_size != 1 else {})

    model = model.eval()
    assert args.world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    tokenizer = llama2_tokenizer(args.model_base, signal_type="chat")
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    text_processor_infer = llama2_text_processor_inference(tokenizer, None, model.image_length)

    file_ = open(args.file_path, 'r')
    entities = file_.read().strip().split("\n")
    file_.close()

    save_file = open(args.save_path, 'a') #"000_deepmeaning_options_cogvlm-prompt3.json"

    for entity in tqdm(entities):

        # load as dict
        triple = json.loads(entity.strip())

        # Image
        image_url = triple['Url']
    
        N = 0
        response = ""
        while ("A\n" not in response) and ("A," not in response) and ("A<" not in response) and ("A." not in response) and ("A)" not in response) and ("B\n" not in response) and ("B," not in response) and ("B<" not in response) and ("B." not in response) and ("B)" not in response) and ("C\n" not in response) and ("C," not in response) and ("C<" not in response) and ("C." not in response) and ("C)" not in response) and ("D\n" not in response) and ("D," not in response) and ("D<" not in response) and ("D." not in response) and ("D)" not in response):
            with torch.no_grad():
                prom = triple[args.prompt].format(triple['Options']['A'],triple['Options']['B'],triple['Options']['C'],triple['Options']['D'])
                response, history, cache_image = chat(
                    image_url, 
                    model, 
                    text_processor_infer,
                    image_processor,
                    prom, 
                    history=[],
                    max_length=args.max_length, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    invalid_slices=text_processor_infer.invalid_slices,
                    no_prompt=False
                    )
                N+=1
                if N>5:
                    break
                
        output = {"Triple":triple, 'Model_Path':args.model_path, "Output":response}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')       

    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_LLaVA-13B.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    main(args)


