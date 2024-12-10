import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer

# modified
from torch.utils.data import Dataset, DataLoader
import traceback
import re
import math

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.') # , required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.') # , required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.') # , required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.') #, required=True)
    parser.add_argument("--model-name", type=str,) # required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, ) #required=True)

    return parser.parse_args()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class PerceptionTestMCQADataset(Dataset):
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    def __init__(self, data_list):
        self.data_list = data_list
        # self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_name = line['metadata']['video_id']
        mc_questions = line['mc_question']

        # for fmt in self.video_formats:  # Added this line
        fmt = ".mp4"  # hard coded for mcqa
        temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
        if os.path.exists(temp_path):
            video_path = temp_path
            video_tensor = load_video(video_path)
        else:
            print(temp_path, "does not exist... returning")
            return None
        instructs = []
        qids = []
        ops = []
        ans = []
        for q in mc_questions:
            question = q['question']
            qid = q['id']
            options = q['options']
            q_ans = q['answer_id']
            instruct = f'Question: {question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'

            instructs.append(instruct)
            qids.append(qid)
            ops.append(options)
            ans.append(q_ans)

        return {
            'video': video_tensor,
            'video_id': video_name,
            'instructs': instructs,
            'question_ids': qids,
            'options': ops,
            'answer': ans,
        }


def collate_fn(batch):
    vid = [x['video'] for x in batch]
    v_id = [x['video_id'] for x in batch]
    ins = [x['instructs'] for x in batch]
    q_ids = [x['question_ids'] for x in batch]
    ops = [x['options'] for x in batch]
    ans = [x['answer'] for x in batch]
    # vid = torch.stack(vid, dim=0)
    return vid, v_id, ins, q_ids, ops, ans


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    Example:
        python video_chatgpt/eval/run_inference_benchmark_general.py \
            --video_dir <path-to-directory-containing-videos> \
            --gt_file <ground-truth-file-containing-question-answer-pairs> \
            --output_dir <output-dir-path> \
            --output_name <output-file-name> \
            --model-name <path-to-LLaVA-Lightening-7B-v1-1> \
            --projection_path <path-to-Video-ChatGPT-weights>
    """
    args.video_dir = "/home/jim/Documents/Projects/perception_test/baselines/data/videos"
    args.gt_file = "/home/jim/Documents/Projects/perception_test/baselines/data/mc_question_train.json"
    args.model_name = "/home/jim/Documents/Projects/Video-ChatGPT/scripts/LLaVA-7B-Lightening-v1-1/"
    args.projection_path = "/home/jim/Documents/Projects/Video-ChatGPT/video_chatgpt-7B.bin"
    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load the question file
    questions = json.load(open(args.gt_file, "r"))
    questions = list(questions.values())
    # questions = get_chunk(questions, 1, 0)  # (questions, num-chunks=1, chunk-idx=0)

    # assert args.batch_size == 1, "Batch size must be 1 for inference"
    batch_size = 1
    num_workers = 4
    dataset = PerceptionTestMCQADataset(questions)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate_fn)

    answer_file = os.path.expanduser("./answer_file.json")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    MAX_EVAL_SIZE = 250

    # Iterate over each sample in the ground truth file
    for i, (video_tensor, video_id, instructs, question_ids, options, true_ans) in enumerate(tqdm(dataloader)):
        # reduce batch dimension
        video_frames = video_tensor[0]
        video_id = video_id[0]
        instructs = instructs[0]
        question_ids = question_ids[0]
        options = options[0]
        true_ans = true_ans[0]
        output = None
        qas = []
        # video_name = video_id
        # sample_set = sample
        # question = sample['Q']
        for idx, instruct in enumerate(instructs):
            letters = ['(A)', '(B)', '(C)']
            question_id = question_ids[idx]
            _options = options[idx]
            q_answer = true_ans[idx]
            # print("Q:", instruct)
            try:
                # Run inference on the video and add the output to the list
                output = video_chatgpt_infer(video_frames, instruct, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
                # print("A: ", output)
            except Exception as e:
                print(f"Error processing video file '{video_id}': {e}")

            output = output.replace('answer', '')
            output = output.replace('Answer', '')
            pred_answer = re.findall('\(*[A-C]\)*', output)
            try:
                assert len(
                    pred_answer) >= 1, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(
                    video_id, instruct, output)
                pred_answer = pred_answer[0].strip()
                # if not pred_answer.startswith('('):
                pred_answer = pred_answer.strip('()')
                pred_answer = f'({pred_answer})'
                pred_idx = letters.index(pred_answer)
            except:
                traceback.print_exc()
                tmp_options = [x.lower() for x in _options]
                if output.lower() in tmp_options:
                    tmp_options = [x.lower() for x in _options]
                    pred_idx = tmp_options.index(output.lower())
                else:
                    pred_idx = 2

            qas.append({'id': question_id, 'question': instruct,'answer_id': pred_idx, 'answer': _options[pred_idx], 'answer_text': output,
                        'true_answer': q_answer, "correct": pred_idx == q_answer})

        ans_file.write('\"{}\": {},\n'.format(video_id, json.dumps(qas)))

        if i > min(MAX_EVAL_SIZE, len(dataloader)):
            print("max eval size reached, exiting...")
            break
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
