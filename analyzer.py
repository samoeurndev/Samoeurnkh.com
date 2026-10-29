import google.generativeai as genai
import os
import time

def analyze_video_to_story(video_path, api_key):
    """
    Uploads a video to Gemini and asks it to summarize the story in Khmer.
    """
    genai.configure(api_key=api_key)
    
    print(f"Uploading video: {video_path}...")
    import subprocess
    import shutil
    optimized_path = f"{video_path}.opt.mp4"
    print("Compressing video locally to prevent Gemini upload timeout...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "scale=-2:480", "-r", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "64k",
        optimized_path
    ]
    try:
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            upload_target = optimized_path
        else:
            upload_target = video_path
    except Exception:
        upload_target = video_path

    video_file = genai.upload_file(path=upload_target)
    
    if upload_target == optimized_path:
        try:
            import os
            os.remove(optimized_path)
        except Exception:
            pass
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("Processing video...")
        time.sleep(5)
        video_file = genai.get_file(video_file.name)
        
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")

    print("Analyzing story...")
    model = genai.GenerativeModel(model_name="gemini-flash-latest")
    
    prompt = """
    # SYSTEM ROLE
    You are an expert Movie Recap Narrator and Storyteller. Your task is to analyze the provided video and generate a highly detailed, continuous narrative recap script in Khmer, timed perfectly to the visual scenes.

    Your output must flow naturally like a popular "movie recap" YouTube video, drawing the audience in while staying perfectly synced to the onscreen action.

    ---

    ## CRITICAL EXECUTION RULES

    ### 1. Continuous, Gapless Speech (No Stops)
    *   The narration must be **CONTIGUOUS** from the beginning to the end of the video. Create a sequence of timestamps where the end time of one segment is exactly the start time of the next (e.g., `[0, 5]`, `[5, 12]`, `[12, 18]`). 
    *   Do not leave any silent gaps or skipped time. Every second of the video must be covered by your story.

    ### 2. Perfect Detail & Meaningful Storytelling
    *   Provide extremely rich and detailed descriptions of the visual events. 
    *   Connect ideas smoothly between segments so it sounds like a human telling an exciting, meaningful story. Include emotional context, drama, and character motivations based on their actions.

    ### 3. Granular Alignment & Pacing
    *   Break the video down into highly-focused segments (typically 4 to 10 seconds each).
    *   To ensure the narration matches the exact video segment duration perfectly, you must apply the correct pacing metric for **Khmer**. 
    *   Pace at exactly **12 to 15 characters per second**.
    *   *[Self-Correction Formula]:* Target Length = (end_time - start_time) × 13.
        *   *Example (8s scene):* 8 × 13 = ~104 characters.

    ---

    ## MANDATORY OUTPUT FORMAT
    Do not include conversational filler, introductory text, or markdown code blocks. Output your response EXACTLY like this, line-by-line:

    [start_time, end_time] Engaging, highly detailed narrative recap text in Khmer here matching the exact pace.
    [start_time, end_time] The story continues smoothly and continuously in Khmer here without any time gaps.

    ### Example (Khmer):
    [0, 8] រឿងនេះចាប់ផ្តើមឡើងនៅពេលដែលបុរសម្នាក់បានដើរចូលទៅក្នុងព្រៃដ៏អាថ៌កំបាំងមួយ...
    [8, 15] ភ្លាមៗនោះ គាត់ក៏បានប្រទះឃើញសត្វដំរីដ៏ធំមួយក្បាលដែលកំពុងតែដេកលក់យ៉ាងស្ងៀមស្ងាត់...

    ---

    ## QUALITY ASSURANCE
    Ensure the script covers every single second of the video continuously, perfectly paced, and feels like an entertaining, deeply detailed story recap!
    """
    
    response = model.generate_content([video_file, prompt])
    
    return response.text

def generate_video_hooks(script, api_key, language="Khmer"):
    """
    Generates engaging video hook suggestions in a structured JSON string format.
    """
    genai.configure(api_key=api_key)
    
    print("Generating video hooks...")
    model = genai.GenerativeModel(model_name="gemini-flash-latest")
    
    prompt = f"""
    Based on the following movie storytelling script in {language}, generate 4 different, highly-engaging video hooks (intro voice-overs) in {language}, along with the start timestamp of the most visually exciting/matching scene in the movie to serve as the visual hook background.

    Provide exactly 4 options with the following styles:
    1. Curiosity/Clickbait (Make the viewer need to know what happens)
    2. Dramatic/Suspense (Highlight the highest stakes/drama of the story)
    3. Question (Ask an intriguing question about the plot)
    4. Bold Statement/Action (Start with a strong, shocking declaration)

    Format your response EXACTLY as a JSON object with this schema:
    - curiosity: {{ "text": string (The hook text in {language}), "visual_start": number (The recommended start time in seconds of the best matching scene in the movie to show visually) }}
    - dramatic: {{ "text": string (The hook text in {language}), "visual_start": number (The recommended start time in seconds of the best matching scene in the movie to show visually) }}
    - question: {{ "text": string (The hook text in {language}), "visual_start": number (The recommended start time in seconds of the best matching scene in the movie to show visually) }}
    - bold: {{ "text": string (The hook text in {language}), "visual_start": number (The recommended start time in seconds of the best matching scene in the movie to show visually) }}

    Select "visual_start" based on the timestamps in the story script below, finding the most engaging/relevant scene for each hook's specific tone.
    Each hook text MUST be concise (under 120 characters) and extremely punchy. Do not include timestamp markers or quotes around the hook itself.
    
    STORY SCRIPT:
    {script}
    """
    
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    return response.text

if __name__ == "__main__":
    # To use this, you need a Gemini API Key from: https://aistudio.google.com/
    MY_API_KEY = "[ENCRYPTION_KEY]"
    VIDEO_PATH = "sample_video.mp4" # Replace with your video
    
    if MY_API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your Gemini API Key in analyzer.py")
    elif not os.path.exists(VIDEO_PATH):
        print(f"Video file not found: {VIDEO_PATH}")
    else:
        khmer_script = analyze_video_to_story(VIDEO_PATH, MY_API_KEY)
        
        with open("generated_story.txt", "w", encoding="utf-8") as f:
            f.write(khmer_script)
        
        print("\nScript saved to generated_story.txt")
        print("Content saved successfully to file.")
        
        try:
            hooks_json = generate_video_hooks(khmer_script, MY_API_KEY)
            with open("generated_hooks.json", "w", encoding="utf-8") as f:
                f.write(hooks_json)
            print("\nHooks saved to generated_hooks.json")
            print("--- Generated Hooks ---")
            print(hooks_json)
        except Exception as e:
            print(f"Could not generate hooks: {e}")

