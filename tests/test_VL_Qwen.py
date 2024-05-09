from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
from tqdm import tqdm


def main(args):
    torch.manual_seed(1234)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device, trust_remote_code=True).eval()

    # file loading
    file_ = open(args.file_path, 'r')
    entities = file_.read().strip().split("\n")
    file_.close()

    save_file = open(args.save_file, 'a')

    for entity in tqdm(entities):
    # load as dict
        triple = json.loads(entity.strip())

        # Image
        image_url = triple['Url']

        # Prompt
        inp =  triple[arg.prompt].format(triple['options']['A'],triple['options']['B'],triple['options']['C'],triple['options']['D'])

        query = tokenizer.from_list_format([
            {'image': image_url},
            {'text': inp}
        ])


        response, history = model.chat(tokenizer, query=query, history=None)
        N = 1
        while ("A\n" not in response) and ("A," not in response) and ("A<" not in response) and ("A." not in response) and ("A)" not in response) and ("B\n" not in response) and ("B," not in response) and ("B<" not in response) and ("B." not in response) and ("B)" not in response) and ("C\n" not in response) and ("C," not in response) and ("C<" not in response) and ("C." not in response) and ("C)" not in response) and ("D\n" not in response) and ("D," not in response) and ("D<" not in response) and ("D." not in response) and ("D)" not in response):          
            response, history = model.chat(tokenizer, query=query, history=None)
            N+=1
            if N>5:  
                break

        output = {"Triple":triple, 'Model_Path':args.model_path, "Output":response}
        sen_dict = json.dumps(output)
        save_file.write(sen_dict + '\n')       

    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen-VL-Chat")
    parser.add_argument("--file-path", type=str, default='./data/DeepSemantics_Questions.json')
    parser.add_argument("--save-path", type=str, default='./result/DeepSemantics_Prompt_1_VL_Qwen.json')
    parser.add_argument("--prompt", type=str, default='Prompt_1')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
