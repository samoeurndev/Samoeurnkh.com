"""
FastAPI backend for Movie Script Analyzer.
Handles video upload to Gemini, script generation, and TTS audio generation.
"""

import asyncio
import os
import re
import time
import tempfile
import struct
import io
import edge_tts
import subprocess
import json
import sys

# Reconfigure stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
import imageio_ffmpeg

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


app = FastAPI(title="Movie Script Analyzer API")

# CORS for Vue.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds using ffmpeg."""
    try:
        cmd = [FFMPEG_EXE, "-i", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?", result.stderr)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            ms_str = match.group(4) or "0"
            ms = float(f"0.{ms_str}")
            duration = hours * 3600 + minutes * 60 + seconds + ms
            return duration
    except Exception as e:
        print(f"DEBUG: Failed to get media duration: {e}")
    return 3.0  # Fallback to 3 seconds if parsing fails


def split_video_into_chunks(source_path: str, max_chunk_duration: float = 240.0) -> list[str]:
    """Split a video into multiple chunk files if it exceeds the max duration."""
    duration = get_audio_duration(source_path)
    chunks = []
    start = 0.0
    index = 0

    while start < duration:
        chunk_length = min(max_chunk_duration, duration - start)
        chunk_path = f"{source_path}.chunk_{index}.mp4"
        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", source_path,
            "-ss", f"{start:.2f}",
            "-t", f"{chunk_length:.2f}",
            "-vf", "scale=-2:480",
            "-r", "2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-c:a", "aac",
            "-b:a", "64k",
            "-avoid_negative_ts", "1",
            chunk_path,
        ]
        print(f"DEBUG: Splitting and compressing chunk {index} ({start:.2f}s to {start + chunk_length:.2f}s): {chunk_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        if result.returncode != 0:
            print(f"DEBUG: Failed to compress video chunk. Fallback to copy. Error: {result.stderr}")
            cmd_fallback = [
                FFMPEG_EXE, "-y", "-i", source_path,
                "-ss", f"{start:.2f}", "-t", f"{chunk_length:.2f}",
                "-c", "copy", "-avoid_negative_ts", "1", chunk_path,
            ]
            subprocess.run(cmd_fallback)
        chunks.append(chunk_path)
        index += 1
        start += max_chunk_duration

    return chunks


def normalize_script_timestamps(script: str, offset: float) -> str:
    """Adjust timestamp ranges to the full video timeline using a segment offset."""
    if offset <= 0:
        return script

    adjusted_lines = []
    should_adjust = False

    for line in script.splitlines():
        match = re.match(r"^\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]\s*(.*)$", line)
        if match:
            start = float(match.group(1))
            if start + 1e-6 < offset:
                should_adjust = True
                break

    if not should_adjust:
        return script

    for line in script.splitlines():
        match = re.match(r"^\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]\s*(.*)$", line)
        if match:
            start = float(match.group(1)) + offset
            end = float(match.group(2)) + offset
            text = match.group(3)
            adjusted_lines.append(f"[{start:.2f}, {end:.2f}] {text}")
        else:
            adjusted_lines.append(line)

    return "\n".join(adjusted_lines)


def cleanup_uploads(max_files: int = 10):
    """Keep the uploads directory clean to prevent 'No space left on device' errors."""
    try:
        if not os.path.exists("uploads"):
            return
        files = [os.path.join("uploads", f) for f in os.listdir("uploads")]
        # Sort by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        
        # If we have more than max_files, delete the oldest ones
        if len(files) > max_files:
            for f in files[:-max_files]:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                        print(f"DEBUG: Cleaned up old file: {f}")
                except Exception as e:
                    print(f"DEBUG: Failed to delete {f}: {e}")
    except Exception as e:
        print(f"DEBUG: Cleanup error: {e}")


@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    api_key: str = Form(...),
    language: str = Form(default="Khmer"),
    model_name: str = Form(default="gemini-1.5-flash"),
):
    """
    Upload a video, send it to Gemini for analysis, and return a timestamped
    storytelling script.
    """
    if not api_key.strip():
        raise HTTPException(status_code=400, detail="API key is required")

    # Cleanup old uploads to free space
    cleanup_uploads(max_files=5)

    # Validate file type
    content_type = video.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    # Save uploaded file to the uploads directory
    # Ensure video_id is just a filename, not a full path
    raw_filename = os.path.basename(video.filename or "video.mp4")
    # Strip ALL non-ASCII characters from the filename so the Gemini API HTTP
    # client (which encodes headers as ASCII) never encounters a UnicodeEncodeError.
    # Example: "4K ᴴᴰ.mp4" → "4K .mp4" → "4K.mp4"
    name_part, ext_part = os.path.splitext(raw_filename)
    name_ascii = re.sub(r'[^\x00-\x7F]+', '', name_part).strip()  # drop non-ASCII
    name_ascii = re.sub(r'[\s_-]+', '_', name_ascii).strip('_')   # normalise spaces
    safe_filename = (name_ascii or "video") + ext_part
    video_id = f"{int(time.time())}_{safe_filename}"
    video_path = os.path.join("uploads", video_id)

    chunk_paths = []
    try:
        print(f"DEBUG: Saving video to {video_path}")
        with open(video_path, "wb") as f:
            contents = await video.read()
            f.write(contents)

        video_duration = get_audio_duration(video_path)
        print(f"DEBUG: Uploaded video duration: {video_duration:.2f}s")
        print("DEBUG: Compressing and chunking video for fast Gemini upload")
        chunk_paths = split_video_into_chunks(video_path, max_chunk_duration=240.0)

        # Initialize Gemini client with a 30-minute timeout (values are in MILLISECONDS)
        client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(timeout=1800000) # 30 minutes in ms
        )

        prompt_base = f"""
        # SYSTEM ROLE
        You are an expert Movie Recap Narrator and Storyteller. Your task is to analyze the provided video and generate a highly detailed, continuous narrative recap script in {language}, timed perfectly to the visual scenes.

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
        *   To ensure the narration matches the exact video segment duration perfectly, you must apply the correct pacing metric for **{language}**. 
        *   *If {language} is Khmer:* Pace at exactly **12 to 15 characters per second**.
        *   *If {language} is English/Spanish/Western:* Pace at exactly **2.5 to 3 words per second**.
        *   *If {language} is Chinese/Japanese:* Pace at exactly **3 to 4 characters per second**.
        *   *[Self-Correction Formula]:* Target Length = (end_time - start_time) × (Language Pacing Metric). 

        ---

        ## MANDATORY OUTPUT FORMAT
        Do not include conversational filler, introductory text, or markdown code blocks. Output your response EXACTLY like this, line-by-line:

        [start_time, end_time] Engaging, highly detailed narrative recap text in {language} here matching the exact pace.
        [start_time, end_time] The story continues smoothly and continuously in {language} here without any time gaps.

        ### Example (Khmer):
        [0, 8] រឿងនេះចាប់ផ្តើមឡើងនៅពេលដែលបុរសម្នាក់បានដើរចូលទៅក្នុងព្រៃដ៏អាថ៌កំបាំងមួយ...
        [8, 15] ភ្លាមៗនោះ គាត់ក៏បានប្រទះឃើញសត្វដំរីដ៏ធំមួយក្បាលដែលកំពុងតែដេកលក់យ៉ាងស្ងៀមស្ងាត់...

        ---

        ## QUALITY ASSURANCE
        Ensure the script covers every single second of the video continuously, perfectly paced, and feels like an entertaining, deeply detailed story recap!
        """

        script_parts = []
        for idx, chunk_path in enumerate(chunk_paths):
            chunk_offset = idx * 240.0
            upload_path = chunk_path

            print(f"DEBUG: Uploading chunk {idx + 1}/{len(chunk_paths)} to Gemini: {upload_path}")
            video_file = client.files.upload(file=upload_path)

            # Poll until processing completes for this chunk
            max_wait = 1800
            elapsed = 0
            while True:
                try:
                    video_file = client.files.get(name=video_file.name)
                    state = video_file.state
                    if str(state) in ["FileState.ACTIVE", "ACTIVE", "SUCCEEDED"]:
                        print(f"DEBUG: Chunk {idx + 1} processed successfully in {elapsed}s.")
                        break
                    if str(state) in ["FileState.FAILED", "FAILED"]:
                        print(f"DEBUG: Chunk {idx + 1} failed. State: {state}")
                        raise HTTPException(status_code=500, detail=f"Video processing failed on Gemini: {state}")
                    if elapsed % 30 == 0:
                        print(f"DEBUG: Processing chunk {idx + 1}... State: {state} ({elapsed}s / {max_wait}s)")
                    if elapsed >= max_wait:
                        print(f"DEBUG: Chunk {idx + 1} timed out after {max_wait}s")
                        raise HTTPException(status_code=504, detail="Video processing timed out. The file is large and needs more time.")
                    await asyncio.sleep(10)
                    elapsed += 10
                except HTTPException:
                    raise
                except Exception as poll_err:
                    print(f"DEBUG: Chunk polling warning: {poll_err}")
                    await asyncio.sleep(10)
                    elapsed += 10
                    continue

            print(f"DEBUG: Chunk {idx + 1} ready. Waiting 2s for stability...")
            await asyncio.sleep(2)

            chunk_prompt = prompt_base
            if len(chunk_paths) > 1:
                chunk_prompt += f"\n\nNOTE: This is chunk {idx + 1}/{len(chunk_paths)} of the original video. "
                chunk_prompt += f"The full original video starts at {chunk_offset:.2f} seconds for this chunk. "
                chunk_prompt += "All timestamps must be absolute relative to the full original video timeline, not the chunk timeline."

            print(f"DEBUG: Calling Gemini generate_content for chunk {idx + 1}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[video_file, chunk_prompt]
            )

            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass

            if not response.candidates or not response.candidates[0].content.parts:
                raise HTTPException(status_code=500, detail="Gemini returned an empty response. Check safety filters.")

            chunk_script = response.text
            if len(chunk_paths) > 1 and chunk_offset > 0:
                chunk_script = normalize_script_timestamps(chunk_script, chunk_offset)

            script_parts.append(chunk_script.strip())

        script_text = "\n".join(script_parts)

        return JSONResponse(
            content={
                "script": script_text,
                "status": "success",
                "video_id": video_id,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("DEBUG: Analysis failed with CRITICAL error:")
        print(f"ERROR TYPE: {type(e)}")
        print(f"ERROR DETAILS: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in chunk_paths:
            if path != video_path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"DEBUG: Removed temporary chunk file: {path}")
                except Exception as cleanup_err:
                    print(f"DEBUG: Failed to remove chunk file {path}: {cleanup_err}")
    # Note: We are NOT deleting the file here anymore so it can be used for video generation later.


@app.post("/api/generate-video")
async def generate_video(
    video_id: str = Form(...),
    scenes_json: str = Form(...),
    audio: UploadFile = File(None),
    hook_text: str = Form(default=""),
    hook_visual_start: float = Form(default=0.0),
    mode: str = Form(default="script_text"),
    scale_video: bool = Form(default=True),
):
    """
    Generate a new video with an optional introductory "Hook" voice-over or custom audio track.
    """
    video_path = os.path.join("uploads", video_id)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Original video not found. Please re-analyze.")

    try:
        scenes = json.loads(scenes_json) # List of dicts or list of [start, end]
        
        # Save custom/narration audio if uploaded
        narration_path = None
        if audio is not None:
            narration_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            with open(narration_path, "wb") as f:
                contents = await audio.read()
                if contents:
                    f.write(contents)
                else:
                    narration_path = None

        segment_files = []
        temp_audios = []

        if mode == "audio_speech":
            if not narration_path:
                raise HTTPException(status_code=400, detail="Voice recording or custom audio narration is required for Audio Speech mode.")
            
            # Determine custom audio duration
            A_total = get_audio_duration(narration_path)
            print(f"DEBUG: Audio speech duration: {A_total:.2f}s")
            
            # Calculate total original duration of scenes
            V_total = 0.0
            parsed_scenes = []
            for scene in scenes:
                if isinstance(scene, dict):
                    start = float(scene.get("start", 0))
                    end = float(scene.get("end", 0))
                elif isinstance(scene, list) and len(scene) == 2:
                    start = float(scene[0])
                    end = float(scene[1])
                else:
                    continue
                V_len = end - start
                if V_len > 0:
                    V_total += V_len
                    parsed_scenes.append((start, end, V_len))
            
            if V_total <= 0:
                raise HTTPException(status_code=400, detail="No valid scenes selected for dynamic audio alignment.")
            
            print(f"DEBUG: Total original video duration: {V_total:.2f}s")
            
            # Process each scene segment
            for i, (start, end, V_len) in enumerate(parsed_scenes):
                aligned_seg_path = os.path.join("uploads", f"aligned_{i}_{video_id}.mp4")
                
                # Slices the video segment frame-accurately (no audio)
                slice_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                
                if scale_video:
                    # Dynamically scale video pacing to match the speech length
                    # Playback speed factor is speed_ratio = V_total / A_total
                    speed_ratio = V_total / A_total
                    target_len = V_len * (A_total / V_total)
                    print(f"DEBUG: Segment {i} - Scale ratio: {speed_ratio:.6f}, Target duration: {target_len:.2f}s")
                    
                    cmd_slice = [
                        FFMPEG_EXE, "-y",
                        "-ss", f"{start:.2f}",
                        "-i", video_path,
                        "-t", f"{V_len:.2f}",
                        "-filter_complex", f"[0:v]setpts=PTS/{speed_ratio:.6f}[v]",
                        "-map", "[v]",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-an",
                        "-t", f"{target_len:.2f}",
                        slice_video_tmp
                    ]
                else:
                    # Keep original speed
                    cmd_slice = [
                        FFMPEG_EXE, "-y",
                        "-ss", f"{start:.2f}",
                        "-i", video_path,
                        "-t", f"{V_len:.2f}",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                        "-an",
                        slice_video_tmp
                    ]
                    target_len = V_len
                
                subprocess.run(cmd_slice, check=True)
                
                # Pad scaled video with silent audio so it can be safely concatenated
                cmd_merge = [
                    FFMPEG_EXE, "-y",
                    "-i", slice_video_tmp,
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{target_len:.2f}",
                    aligned_seg_path
                ]
                subprocess.run(cmd_merge, check=True)
                segment_files.append(aligned_seg_path)
                try: os.unlink(slice_video_tmp)
                except: pass
            
            if not segment_files:
                raise HTTPException(status_code=400, detail="Failed to slice scene segments.")
                
            # Concatenate the silent speed-adjusted segments
            list_path = os.path.join("uploads", f"list_{video_id}.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for seg in segment_files:
                    seg_safe = os.path.abspath(seg).replace('\\', '/')
                    f.write(f"file '{seg_safe}'\n")

            concat_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            cmd_concat = [
                FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", concat_video_tmp
            ]
            
            print(f"DEBUG: Concatenating {len(segment_files)} segments...")
            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ffmpeg concat failed: {result.stderr}")
                
            # Overlay custom audio track onto concatenated speed-scaled video
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            cmd_merge_final = [
                FFMPEG_EXE, "-y",
                "-i", concat_video_tmp,
                "-i", narration_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", f"{A_total:.2f}",
                output_path
            ]
            
            print("DEBUG: Merging final video with audio narration track...")
            result = subprocess.run(cmd_merge_final, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"ffmpeg final merge failed: {result.stderr}")
                
            try: os.unlink(concat_video_tmp)
            except: pass
            try: os.unlink(list_path)
            except: pass
            for seg in segment_files:
                try: os.unlink(seg)
                except: pass
            if narration_path:
                try: os.unlink(narration_path)
                except: pass

        else:
            # SCRIPT TEXT & AUTO-TTS MODE (original segment-by-segment logic)
            # 1. Compile Aligned Hook Segment
            hook_segment_path = None
            if hook_text.strip():
                print(f"DEBUG: Generating hook audio for: {hook_text}")
                hook_audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                voice = "km-KH-PisethNeural"
                communicate = edge_tts.Communicate(hook_text, voice)
                await communicate.save(hook_audio_tmp)

                hook_duration = get_audio_duration(hook_audio_tmp)
                print(f"DEBUG: Hook audio duration: {hook_duration:.2f}s")

                hook_segment_path = os.path.join("uploads", f"hook_{video_id}.mp4")
                # Step 1: Slice hook video segment first frame-accurately (no audio)
                slice_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                cmd_slice = [
                    FFMPEG_EXE, "-y",
                    "-ss", f"{hook_visual_start:.2f}",
                    "-i", video_path,
                    "-t", f"{hook_duration:.2f}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-an",
                    slice_video_tmp
                ]
                subprocess.run(cmd_slice, check=True)

                # Step 2: Merge hook video with hook audio narration
                cmd_merge = [
                    FFMPEG_EXE, "-y",
                    "-i", slice_video_tmp,
                    "-i", hook_audio_tmp,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", f"{hook_duration:.2f}",
                    hook_segment_path
                ]
                subprocess.run(cmd_merge, check=True)
                segment_files.append(hook_segment_path)
                try: os.unlink(slice_video_tmp)
                except: pass
                
                try: os.unlink(hook_audio_tmp)
                except: pass

            # 2. Compile Aligned Scene Segments
            for i, scene in enumerate(scenes):
                if isinstance(scene, dict):
                    start = float(scene.get("start", 0))
                    end = float(scene.get("end", 0))
                    text = scene.get("text", "").strip()
                elif isinstance(scene, list) and len(scene) == 2:
                    start = float(scene[0])
                    end = float(scene[1])
                    text = ""
                else:
                    continue

                V_len = end - start
                if V_len <= 0:
                    continue

                aligned_seg_path = os.path.join("uploads", f"aligned_{i}_{video_id}.mp4")

                # Step 1: Slice scene video segment first frame-accurately (no audio)
                slice_video_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                cmd_slice = [
                    FFMPEG_EXE, "-y",
                    "-ss", f"{start:.2f}",
                    "-i", video_path,
                    "-t", f"{V_len:.2f}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-an",
                    slice_video_tmp
                ]
                subprocess.run(cmd_slice, check=True)

                if text:
                    print(f"DEBUG: Generating segment {i} audio for text: {text}")
                    scene_audio_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    temp_audios.append(scene_audio_tmp)

                    voice = "km-KH-PisethNeural"
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(scene_audio_tmp)

                    A_len = get_audio_duration(scene_audio_tmp)
                    print(f"DEBUG: Segment {i} - Video len: {V_len:.2f}s, Audio len: {A_len:.2f}s")

                    if A_len <= V_len:
                        # Video original speed. Audio padded with silence at the end.
                        cmd_merge = [
                            FFMPEG_EXE, "-y",
                            "-i", slice_video_tmp,
                            "-i", scene_audio_tmp,
                            "-filter_complex", f"[1:a]apad=pad_dur={V_len}[a]",
                            "-map", "0:v:0", "-map", "[a]",
                            "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k",
                            "-t", f"{V_len:.2f}",
                            aligned_seg_path
                        ]
                    else:
                        # Video slowed down to match audio duration exactly.
                        speed_ratio = V_len / A_len
                        cmd_merge = [
                            FFMPEG_EXE, "-y",
                            "-i", slice_video_tmp,
                            "-i", scene_audio_tmp,
                            "-filter_complex", f"[0:v]setpts=PTS/{speed_ratio:.6f}[v]",
                            "-map", "[v]", "-map", "1:a:0",
                            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k",
                            "-t", f"{A_len:.2f}",
                            aligned_seg_path
                        ]
                else:
                    # Silent scene (pad with synthesized silent audio)
                    print(f"DEBUG: Segment {i} - Silent video segment")
                    cmd_merge = [
                        FFMPEG_EXE, "-y",
                        "-i", slice_video_tmp,
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-t", f"{V_len:.2f}",
                        aligned_seg_path
                    ]

                subprocess.run(cmd_merge, check=True)
                segment_files.append(aligned_seg_path)
                try: os.unlink(slice_video_tmp)
                except: pass

            if not segment_files:
                raise HTTPException(status_code=400, detail="No valid scenes to generate video")

            # 3. Concatenate Aligned Segments
            list_path = os.path.join("uploads", f"list_{video_id}.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for seg in segment_files:
                    seg_safe = os.path.abspath(seg).replace('\\', '/')
                    f.write(f"file '{seg_safe}'\n")

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            cmd_concat = [
                FFMPEG_EXE, "-y", "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", output_path
            ]
            
            log_file = "ffmpeg_debug.log"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- NEW MERGE CONCAT: {time.ctime()} ---\n")
                f.write(f"Concat List: {list_path}\n")

            result = subprocess.run(cmd_concat, capture_output=True, text=True)
            
            with open(log_file, "a") as f:
                f.write("STDOUT:\n" + result.stdout + "\n")
                f.write("STDERR:\n" + result.stderr + "\n")
                f.write(f"RETURN CODE: {result.returncode}\n")

            if result.returncode != 0:
                raise Exception(f"ffmpeg concat failed: {result.stderr}")

            # 4. Cleanups
            try: os.unlink(list_path)
            except: pass
            for seg in segment_files:
                try: os.unlink(seg)
                except: pass
            for aud in temp_audios:
                try: os.unlink(aud)
                except: pass
            if narration_path:
                try: os.unlink(narration_path)
                except: pass

        def iterfile():
            with open(output_path, mode="rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
            try: os.unlink(output_path)
            except: pass

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": 'attachment; filename="narrated_story.mp4"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )

    except Exception as e:
        import traceback
        print("DEBUG: Video generation failed:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download-script")
async def download_script(
    script: str = Form(...),
    filename: str = Form(default="script.txt")
):
    """
    Download the provided script text as a .txt file.
    """
    return StreamingResponse(
        io.BytesIO(script.encode("utf-8")),
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM bytes to a WAV file in memory."""
    data_size = len(pcm_data)
    wav = io.BytesIO()
    # RIFF header
    wav.write(b"RIFF")
    wav.write(struct.pack("<I", 36 + data_size))
    wav.write(b"WAVE")
    # fmt chunk
    wav.write(b"fmt ")
    wav.write(struct.pack("<I", 16))                     # chunk size
    wav.write(struct.pack("<H", 1))                      # PCM format
    wav.write(struct.pack("<H", channels))
    wav.write(struct.pack("<I", sample_rate))
    wav.write(struct.pack("<I", sample_rate * channels * sample_width))  # byte rate
    wav.write(struct.pack("<H", channels * sample_width))               # block align
    wav.write(struct.pack("<H", sample_width * 8))                      # bits per sample
    # data chunk
    wav.write(b"data")
    wav.write(struct.pack("<I", data_size))
    wav.write(pcm_data)
    wav.seek(0)
    return wav.read()


@app.post("/api/generate-audio")
async def generate_audio(
    script: str = Form(...),
    api_key: str = Form(...),
):
    """
    Generate TTS audio from script text using Edge-TTS (high quality, no quota limits).
    Returns an MP3 audio file.
    """
    if not script.strip():
        raise HTTPException(status_code=400, detail="Script text is required")

    try:
        # Extract narration text only from lines that contain timestamp brackets [start, end]
        # This automatically skips intro/outro conversational filler like "ខាងក្រោមនេះគឺជាស្គ្រីបនិទានរឿង..."
        lines = []
        for line in script.splitlines():
            # Check if line contains a timestamp bracket [ ... ]
            if re.search(r"\[[^\]]+\]", line):
                # Strip the timestamp bracket and add the clean text
                clean_line = re.sub(r"\[[^\]]+\]", "", line).strip()
                if clean_line:
                    lines.append(clean_line)
        
        clean_script = " ".join(lines)


        # Define the Khmer voice
        voice = "km-KH-PisethNeural" # High quality male voice

        # Initialize edge-tts
        communicate = edge_tts.Communicate(clean_script, voice)

        async def audio_generator():
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
            except Exception as e:
                print(f"DEBUG: Streaming error: {e}")
                # We can't raise a new HTTPException inside a generator, 
                # but we can log it and stop the stream.
                return

        return StreamingResponse(
            audio_generator(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'attachment; filename="narration.mp3"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            },
        )

    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"DEBUG: Audio generation failed: {err_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=err_msg)


@app.post("/api/generate-seo")
async def generate_seo(
    script: str = Form(...),
    api_key: str = Form(...),
    language: str = Form(default="Khmer"),
    model_name: str = Form(default="gemini-1.5-flash"),
):
    """
    Generate Khmer SEO metadata (Title, Description, Tags) for FB and YouTube.
    """
    if not script.strip():
        raise HTTPException(status_code=400, detail="Script text is required")

    try:
        client = genai.Client(api_key=api_key.strip())
        
        prompt = f"""
        Based on the following movie storytelling script in {language}, generate high-quality SEO metadata 
        for both YouTube and Facebook Page in {language}.
        
        CRITICAL POLICY REQUIREMENTS:
        1. NO CLICKBAIT: Titles must accurately reflect the content.
        2. NO MISLEADING INFORMATION: Descriptions must be truthful.
        3. RELEVANT TAGS: Use only tags directly related to the story.
        4. COMMUNITY GUIDELINES: Ensure content is safe and appropriate.

        Format your response EXACTLY as a JSON object with these keys:
        - youtube_title: string (Catchy but accurate, under 100 chars)
        - youtube_description: string (Engaging summary with keywords)
        - youtube_tags: string (Comma-separated list of relevant tags)
        - facebook_headline: string (Short and punchy)
        - facebook_description: string (Engagement-focused summary)

        STORY SCRIPT:
        {script}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="Gemini returned an empty response.")

        import json
        seo_data = json.loads(response.text)
        
        return JSONResponse(content={"seo": seo_data, "status": "success"})

    except Exception as e:
        import traceback
        print(f"DEBUG: SEO generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-hooks")
async def generate_hooks(
    script: str = Form(...),
    api_key: str = Form(...),
    language: str = Form(default="Khmer"),
    model_name: str = Form(default="gemini-1.5-flash"),
):
    """
    Generate multiple engaging Video Hook options based on the movie storytelling script.
    Categorized by hook styles: Curiosity, Dramatic, Question, Bold.
    """
    if not script.strip():
        raise HTTPException(status_code=400, detail="Script text is required")

    try:
        client = genai.Client(api_key=api_key.strip())
        
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

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="Gemini returned an empty response.")

        hooks_data = json.loads(response.text)
        
        return JSONResponse(content={"hooks": hooks_data, "status": "success"})

    except Exception as e:
        import traceback
        print(f"DEBUG: Hook generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/download-seo")
async def download_seo(
    seo_data: str = Form(...),
    filename: str = Form(default="seo_metadata.txt")
):
    """
    Download the generated SEO metadata as a text file.
    """
    try:
        import json
        data = json.loads(seo_data)
        
        content = f"""--- YOUTUBE SEO ({filename}) ---
TITLE: {data.get('youtube_title', 'N/A')}

DESCRIPTION:
{data.get('youtube_description', 'N/A')}

TAGS:
{data.get('youtube_tags', 'N/A')}

--- FACEBOOK SEO ---
HEADLINE: {data.get('facebook_headline', 'N/A')}

POST DESCRIPTION:
{data.get('facebook_description', 'N/A')}
"""
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to format SEO file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
