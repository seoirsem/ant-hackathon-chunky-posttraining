import argparse
from transformers import pipeline
import torch
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_between_tags(text, tag):
   if not isinstance(text, str) or not isinstance(tag, str):
       return None
   start_tag = f"<{tag}>"
   end_tag = f"</{tag}>"
   start_index = text.find(start_tag)
   if start_index == -1:
       return None
   end_index = text.find(end_tag, start_index)
   if end_index == -1:
       return None
   return text[start_index + len(start_tag):end_index]

def process_in_batches(data, pipeline, batch_size=8, num_batches=10):
    results = []
    for i in tqdm.tqdm(range(0, len(data), batch_size), total=num_batches):
        batch_inputs = [data["generation"][x] for x in range(i, i+batch_size)]
        batch_results = pipeline(batch_inputs)
        results.extend(batch_results)
        if i>=(num_batches-1)*batch_size:
            break
    return results



def eval(model_path, data_path, task_path, work_dir, batch_size, num_batches):
    pipeline_test = pipeline(
        task="text-generation",
        model=str(model_path),
        torch_dtype=torch.float16,
        device=device
    )
    dataset, task_description = get_dataset(data_path, task_path)
    processed_data = process_in_batches(data, pipeline_test, batch_size, num_batches)

    results = {"straight": 0, "cross": 0, "count": 0}
    for idx, result in enumerate(processed_data):
        if data[idx]['task'] == 'task_a':
            label = "reddit"
            gt = extract_text_between_tags(data[idx]['label'], label)
            # generated_text_story = extract_text_between_tags(result[0]["generated_text"], "story")
            straight_result = extract_text_between_tags(result[0]["generated_text"], label)
            cross_result = extract_text_between_tags(processed_data[idx][0]["generated_text"], label)
            if gt == straight_result:
                results["straight"] += 1
            if gt == cross_result:
                results["cross"] += 1
            results["count"] += 1

    print(f"Results: {results}")

    return data





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--task_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_batches", type=int, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    eval(args.model_path, args.data_path, args.task_path, args.work_dir, args.batch_size, args.num_batches)





