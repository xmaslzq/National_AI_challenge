# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import av
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# Load dataset
df = pd.read_parquet("data/test-00000-of-00001.parquet")

# Load model and processor
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Read video frames with PyAV
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

# Store results
results = []

# Process each row
for idx, row in df.iloc[700:].iterrows():
    video_id = row['video_id']
    question = row['question']
    prompt = row['question_prompt']
    qid = row['qid']

    video_path = f"data/{video_id}.mp4"
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found! Skipping...")
        continue

    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        clip = read_video_pyav(container, indices)
        

        # Prepare prompts
        standard_prompt = f"USER: <video> {question} {prompt} ASSISTANT:"
        devil_prompt = f"USER: <video> {question} {prompt} As a Devil's Advocate, review again and provide the final answer. ASSISTANT:"
        
        # Process inputs
        inputs_std = processor(text=standard_prompt, videos=clip, return_tensors="pt")
        inputs_da = processor(text=devil_prompt, videos=clip, return_tensors="pt")

        # Generate outputs
        std_ids = model.generate(**inputs_std, max_new_tokens=80)
        da_ids = model.generate(**inputs_da, max_new_tokens=80)
        
        std_answer = processor.batch_decode(std_ids, skip_special_tokens=True)[0]
        da_answer = processor.batch_decode(da_ids, skip_special_tokens=True)[0]

        def clean(text):
                if "ASSISTANT:" in text:
                    text = text.split("ASSISTANT:")[-1]
                return text.strip()
        # Store result
        results.append({
            "qid": qid,
            #"video_id": video_id,
            #"question": question,
            #"standard_answer": clean(std_answer),
            "devils_advocate_answer": clean(da_answer) #we changed the title to pred to fit the example excel file.
        })

        print(f"[?] Processed QID: {qid}")

    except Exception as e:
        print(f"[!] Error processing QID {qid}: {e}")
        continue

# Save to Excel
output_df = pd.DataFrame(results)
output_df.to_excel("video_llava_responses.xlsx", index=False)
print("\n? All responses saved to video_llava_responses.xlsx")
