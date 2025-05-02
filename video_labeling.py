import os
import time
import tempfile
from typing import Dict, List, Optional, Union, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import cv2
from pathlib import Path
import requests
import json
import sys
from urllib.parse import urlparse
import shutil
from dotenv import load_dotenv


def analyze_media_with_gemini(
        file_path: str,
        api_key: str,
        detail_level: str = "medium",
        extract_timestamps: bool = True,
        frame_sample_rate: int = 10,
        custom_prompt: Optional[str] = None
) -> Dict:
    """
    Analyze media (image or video) using Gemini Flash and return detailed description.

    Args:
        file_path: Path to the media file (image or video)
        api_key: Google Gemini API key
        detail_level: Level of detail for description - "low", "medium", or "high"
        extract_timestamps: For videos, whether to extract descriptions at different timestamps
        frame_sample_rate: For videos, extract a frame every N seconds
        custom_prompt: Optional custom prompt to override default prompts

    Returns:
        Dictionary containing:
        - description: Detailed description of the media
        - keywords: List of classification keywords
        - media_type: "image" or "video"
        - duration: For videos, the duration in seconds
        - processing_time: Time taken to process in seconds
        - timestamps: For videos, descriptions at different timestamps (if extract_timestamps=True)
    """
    start_time = time.time()

    # Initialize Gemini
    genai.configure(api_key=api_key)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()

    # Prepare result dictionary
    result = {
        "description": "",
        "keywords": [],
        "media_type": "",
        "processing_time": 0,
        "timestamps": {}
    }

    # Generate the appropriate prompt based on detail level
    prompt_template = _generate_prompt_template(detail_level, custom_prompt)

    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        # Process image
        result["media_type"] = "image"
        result.update(_process_image(file_path, prompt_template))

    elif file_ext in ['.mp4', '.mov', '.avi', '.mkv']:
        # Process video
        result["media_type"] = "video"
        result.update(_process_video(
            file_path,
            prompt_template,
            extract_timestamps,
            frame_sample_rate
        ))

    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    # Calculate processing time
    result["processing_time"] = time.time() - start_time

    return result


def _generate_prompt_template(detail_level: str, custom_prompt: Optional[str] = None) -> str:
    """Generate appropriate prompt template based on detail level"""

    if custom_prompt:
        return custom_prompt

    if detail_level == "low":
        return """
        Provide a brief description of this media in 1-2 sentences.
        Include only the most prominent subjects, objects, or actions.
        End with a comma-separated list of 3-5 keywords.
        """

    elif detail_level == "medium":
        return """
        Describe this media content in detail, including:
        - Main subjects and their relationships
        - Setting, environment, and atmosphere
        - Colors, lighting, and composition
        - Any text visible in the content
        - Key actions or events (for video)

        Then provide:
        1. A list of 5-7 specific keywords for classification
        2. A short categorical label (e.g. "landscape photography", "product demo video")
        """

    elif detail_level == "high":
        return """
        Provide an extremely detailed analysis of this media, including:

        VISUAL ELEMENTS:
        - Comprehensive inventory of all visible objects, people, text, and elements
        - Precise spatial relationships between elements
        - Detailed color palette analysis and lighting conditions
        - Technical aspects (composition, camera angles, editing techniques)

        CONTEXT & INTERPRETATION:
        - Inferred purpose or intent of the content
        - Emotional tone and atmosphere
        - Cultural references or significance
        - Target audience analysis

        CLASSIFICATION:
        - 10-15 specific keywords for precise classification
        - 3-5 broader category labels
        - Content rating assessment (G, PG, PG-13, R)
        - Content warnings for any potentially sensitive material

        For videos, include specific timestamps for scene changes or notable events.
        """
    else:
        raise ValueError("detail_level must be 'low', 'medium', or 'high'")


def _process_image(file_path: str, prompt: str) -> Dict:
    """Process a single image with Gemini"""

    # Load the model - UPDATED MODEL NAME
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
    )

    try:
        # Open the image
        image = Image.open(file_path)

        # Generate content
        response = model.generate_content([prompt, image])

        # Extract description and keywords
        full_description = response.text

        # Simple keyword extraction (improve with regex for more complex prompts)
        keywords = []
        lines = full_description.split('\n')
        for line in lines:
            if ':' in line and ('keywords' in line.lower() or 'tags' in line.lower()):
                keyword_text = line.split(':', 1)[1].strip()
                keywords = [k.strip() for k in keyword_text.split(',')]
                break

        # If no keywords found with the above method, try extracting from the last line
        if not keywords and len(lines) > 1:
            last_line = lines[-1].strip()
            if ',' in last_line and len(last_line.split(',')) >= 3:
                keywords = [k.strip() for k in last_line.split(',')]

        return {
            "description": full_description,
            "keywords": keywords
        }
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return {
            "description": f"Error processing image: {str(e)}",
            "keywords": []
        }


def _process_video(
        file_path: str,
        prompt: str,
        extract_timestamps: bool = True,
        frame_sample_rate: int = 10
) -> Dict:
    """Process a video by extracting frames and analyzing them"""

    # Open the video
    video = cv2.VideoCapture(file_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open video: {file_path}")

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()

    result = {
        "duration": duration,
        "fps": fps,
        "timestamps": {}
    }

    try:
        frames_to_extract = []
        key_frames = []

        if extract_timestamps:
            # Extract frames at regular intervals
            # Calculate interval between frames
            interval_frames = int(fps * frame_sample_rate)

            for frame_idx in range(0, frame_count, interval_frames):
                # Set video to the frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()

                if success:
                    timestamp = frame_idx / fps
                    # Save the frame
                    frame_path = os.path.join(temp_dir, f"frame_{timestamp:.2f}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_to_extract.append((timestamp, frame_path))

        # Extract key frames for overall description
        # First frame
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video.read()
        if success:
            first_frame_path = os.path.join(temp_dir, "first_frame.jpg")
            cv2.imwrite(first_frame_path, frame)
            key_frames.append(first_frame_path)

        # Middle frame
        middle_frame_idx = frame_count // 2
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        success, frame = video.read()
        if success:
            middle_frame_path = os.path.join(temp_dir, "middle_frame.jpg")
            cv2.imwrite(middle_frame_path, frame)
            key_frames.append(middle_frame_path)

        # Last frame (a bit before the end to avoid blank frames)
        last_frame_idx = int(frame_count * 0.95)
        video.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
        success, frame = video.read()
        if success:
            last_frame_path = os.path.join(temp_dir, "last_frame.jpg")
            cv2.imwrite(last_frame_path, frame)
            key_frames.append(last_frame_path)

        # Close video first before processing frames to avoid file lock issues
        video.release()

        # Process key frames for timestamps
        if extract_timestamps:
            for timestamp, frame_path in frames_to_extract:
                try:
                    timestamp_prompt = f"{prompt}\n\nThis is a frame from a video at timestamp {timestamp:.2f} seconds."
                    frame_result = _process_image(frame_path, timestamp_prompt)
                    result["timestamps"][f"{timestamp:.2f}"] = frame_result
                except Exception as e:
                    print(f"Error processing frame at {timestamp:.2f}s: {str(e)}")
                    result["timestamps"][f"{timestamp:.2f}"] = {
                        "description": f"Error processing frame: {str(e)}",
                        "keywords": []
                    }

        # Create a comprehensive description based on key frames
        video_prompt = f"""
        {prompt}

        This is a video that is {duration:.2f} seconds long.
        Provide a comprehensive description of the video based on these key frames.
        Include information about what might be happening between these frames.
        """

        # Process the key frames - just use the first one for simplicity
        if key_frames and os.path.exists(key_frames[0]):
            try:
                overall_result = _process_image(key_frames[0], video_prompt)
                result.update(overall_result)
            except Exception as e:
                print(f"Error processing overall video description: {str(e)}")
                result.update({
                    "description": f"Error processing video description: {str(e)}",
                    "keywords": []
                })

    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        if video.isOpened():
            video.release()
    finally:
        # Clean up temp directory - use safer approach
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {str(e)}")

    return result


def download_video_from_url(url, output_path):
    """
    Download a video from a URL and save it to the specified path

    Args:
        url: URL of the video to download
        output_path: Path where the video will be saved

    Returns:
        Path to the downloaded video
    """
    print(f"Downloading video from {url}...")

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Save the video to the specified path
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Video downloaded successfully to {output_path}")
    return output_path


def main():
    # Video URL to download and analyze
    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

    # Create outputs directory if it doesn't exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Get the filename from the URL
    parsed_url = urlparse(video_url)
    video_filename = os.path.basename(parsed_url.path)
    video_path = output_dir / video_filename

    try:
        # Check if the video already exists, if not download it
        if not os.path.exists(video_path):
            downloaded_video_path = download_video_from_url(video_url, video_path)
        else:
            downloaded_video_path = video_path
            print(f"Using existing video file at {video_path}")

        # Get Gemini API key from environment variable or prompt user
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = input("Please enter your Gemini API key: ")

        print(f"Analyzing video {video_filename}...")

        # Analyze the video
        result = analyze_media_with_gemini(
            file_path=str(downloaded_video_path),
            api_key=api_key,
            detail_level="high", # medium, low
            frame_sample_rate=10  # Sample a frame every 10 seconds
        )

        # Save the analysis results to a JSON file
        result_path = output_dir / f"{os.path.splitext(video_filename)[0]}_analysis.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Analysis complete! Results saved to {result_path}")

        # Print some information about the results
        print("\nVideo Analysis Summary:")
        print(f"- Duration: {result['duration']:.2f} seconds")
        print(f"- Processing time: {result['processing_time']:.2f} seconds")
        print(f"- Keywords: {', '.join(result['keywords'])}")
        print(f"- Frames analyzed: {len(result['timestamps'])}")
        print(f"\nDescription: {result['description'][:300]}...\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
