import json
import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from tqdm import tqdm
import urllib.request

def main(args):
    #model_path = '/home/yyx/.cache/modelscope/hub/damo/mPLUG-Owl2'
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, load_8bit=args.load_8bit, load_4bit=args.load_8bit, device=args.device)

    file_ = open(args.file_path, 'r')
    entities = file_.read().strip().split("\n")
    file_.close()

    save_file = open(args.save_path, 'a')

    for entity in tqdm(entities):
        # load as dict
        triple = json.loads(entity.strip())

        # Image
        image_url = triple['Url']
        urllib.request.urlretrieve(image_url, 'image_name')
        image =  Image.open('image_name').convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles
        query =  triple[arg.prompt].format(triple['options']['A'],triple['options']['B'],triple['options']['C'],triple['options']['D'])
        inp = DEFAULT_IMAGE_TOKEN + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
        N=0
        sentence = ""
        while ("A\n" not in sentence) and ("A," not in sentence) and ("A<" not in sentence) and ("A." not in sentence) and ("A)" not in sentence) and ("A" not in sentence) and ("B\n" not in sentence) and ("B," not in sentence) and ("B<" not in sentence) and ("B." not in sentence) and ("B)" not in sentence) and ("B" not in sentence) and ("C\n" not in sentence) and ("C," not in sentence) and ("C<" not in sentence) and ("C." not in sentence) and ("C)" not in sentence) and ("C" not in sentence) and ("D\n" not in sentence) and ("D," not in sentence) and ("D<" not in sentence) and ("D." not in sentence) and ("D)" not in sentence) and ("D" not in sentence):
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    #streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            sentence = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            N+=1
            if N>5:
                break
    
        output = {"Triple":triple, 'Model_Path':args.model_path, "Output":sentence}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')       

    save_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_mPlug_Owl2.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
