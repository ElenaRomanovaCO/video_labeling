import os
import time
import mimetypes
from typing import Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pathlib import Path
import requests
import json
import sys
from urllib.parse import urlparse
from dotenv import load_dotenv


def analyze_video_with_gemini(
        file_path: str,
        api_key: str,
        detail_level: str = "medium",
        custom_prompt: Optional[str] = None
) -> Dict:
    """
    Analyze video using Gemini Flash 2.0 and return detailed description.

    Args:
        file_path: Path to the video file
        api_key: Google Gemini API key
        detail_level: Level of detail for description - "low", "medium", or "high"
        custom_prompt: Optional custom prompt to add to detail level prompt

    Returns:
        Dictionary containing detailed analysis of the video
    """
    start_time = time.time()

    # Initialize Gemini with Client approach
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()

    # Verify it's a video file
    supported_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.mpeg', '.flv', '.mpg', '.webm', '.wmv', '.3gpp']
    if file_ext not in supported_extensions:
        raise ValueError(f"Unsupported file type: {file_ext}. Only video files are supported.")

    # Prepare result dictionary
    result = {
        "captions_summary": "",  # short summary of spoken/narrated messaging
        "description": "",  # detailed visual, emotional, and purpose narrative
        "captions": {},  # timestamped phrases shown/spoken
        "keywords": [],  # search terms (tags)
        "media_type": "video",  # always "video"
        "processing_time": 0,  # will be set dynamically
        "timestamps": {},  # timestamp → visual scene description

        # Marketing-specific fields:
        "product_name": "",  # name of the product or service
        "brand": "",  # brand or company name
        "department": "",  # internal function responsible or targeted
        "product_features": [],  # key features or benefits mentioned
        "unique_selling_points": [],  # reasons to choose this product/service
        "call_to_action": ""  # what the viewer is encouraged to do
    }

    # Process video directly with Gemini 2.0 Flash
    try:
        # Upload the file using the File API
        print(f"Uploading video file {file_path}...")
        file_obj = genai.upload_file(path=file_path)
        print(f"File uploaded with name: {file_obj.name}")

        # Wait for the file to be processed - this is critical for videos
        print("Waiting for file to be processed", end="")
        while file_obj.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file_obj = genai.get_file(file_obj.name)
        print()  # New line after processing dots

        # Check if the file processing was successful
        if file_obj.state.name != "ACTIVE":
            raise ValueError(f"File processing failed with state: {file_obj.state.name}")

        print(f"File processed successfully! State: {file_obj.state.name}")

        # Initialize the model
        print("Analyzing video with Gemini Flash 2.0...")
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Create system prompt with structure requirements
        system_prompt = """
        You are a professional video analyst specializing in marketing content. 

        Analyze the following video and extract detailed structured metadata.
        Your output must be a valid JSON object with the following structure:

        {
          "captions_summary": "A 2-3 sentence summary of spoken/narrated content",
          "description": "Detailed paragraph about visuals, mood, and purpose",
          "captions": {"0:05": "caption at 5 seconds", "0:10": "caption at 10 seconds"},
          "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", 
                      "keyword6", "keyword7", "keyword8", "keyword9", "keyword10"],
          "media_type": "video",
          "timestamps": {"0:00": "description of scene at start", "0:05": "description at 5 seconds"},
          "product_name": "Name of product/service",
          "brand": "Brand name",
          "department": "Marketing department responsible",
          "product_features": ["feature1", "feature2", "feature3"],
          "unique_selling_points": ["point1", "point2"],
          "call_to_action": "What viewer should do next"
        }
        """

        # Get the detail level template
        detail_template = _get_detail_template(detail_level)

        # Debug: Print the prompts being used
        print("\n--- SYSTEM PROMPT ---")
        print(system_prompt)

        # Combine detail template with custom prompt if provided
        user_prompt = detail_template
        if custom_prompt:
            user_prompt = f"""
            {detail_template}

            Additional custom instructions:
            {custom_prompt}
            """

        print("\n--- USER PROMPT ---")
        print(user_prompt)

        # Generate content using system and user prompts
        try:
            print("\nTrying role-based content generation...")
            # First approach: Use the conversation format with roles
            response = model.generate_content(
                [
                    {"role": "system", "parts": [system_prompt]},
                    {"role": "user", "parts": [user_prompt, file_obj]}
                ]
            )
            print("Role-based content generation successful!")
        except Exception as model_error:
            print(f"Warning: Role-based content generation failed: {str(model_error)}")
            print("Falling back to standard content generation...")

            # Fallback approach: Combine both prompts if role-based approach fails
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            print("\n--- COMBINED PROMPT ---")
            print(combined_prompt[:500] + "..." if len(combined_prompt) > 500 else combined_prompt)
            response = model.generate_content([combined_prompt, file_obj])

        # Extract response text
        result_text = response.text

        # Try to parse as JSON
        try:
            # Remove any markdown code block indicators if present
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "", 1)
            if result_text.endswith("```"):
                result_text = result_text.replace("```", "", 1)

            result_text = result_text.strip()
            result_json = json.loads(result_text)
            result.update(result_json)
        except json.JSONDecodeError:
            print("Warning: Could not parse response as JSON, returning raw text")
            result["description"] = result_text

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        result["description"] = f"Error processing video: {str(e)}"

    # Calculate processing time
    result["processing_time"] = time.time() - start_time

    return result


def _get_detail_template(detail_level: str) -> str:
    """Get the template for a specific detail level without checking for custom prompt"""

    if detail_level == "low":  # Short-Form Metadata for Quick
        return """
        You are analyzing a short marketing video to create minimal metadata optimized for search and categorization.

        Your output must be a **valid JSON object**, formatted as follows:

        - summary: A 1–2 sentence description of the most visible or important content (product, brand, message).
        - keywords: A flat list of 3–5 simple keyword strings a user might search to find this video.

        DO NOT include markdown, commentary, or explanation. Return ONLY a clean JSON object.

         Follow this example format **exactly** (no markdown, no quotes around field names, and clean plain text values only):
         Example format:

        {
          "summary": "A young woman jogs through a neon-lit city wearing glowing sneakers. The ad promotes a tech-forward fitness lifestyle.",
          "keywords": ["neon sneakers", "fitness ad", "lifestyle brand", "urban running", "wearable tech"]
        }
        DO NOT include text before or after the JSON. Output only a clean JSON object in the format above.
        """

    elif detail_level == "medium":
        return """
        You are analyzing a marketing video to extract structured metadata that supports user-driven search, filtering, and categorization.

        Return your output as a **clean, valid JSON object**. Do not include markdown, backticks, or explanations. Output only the JSON structure described below.

        Your response must include the following fields:
        - product: Name of the main product or service shown
        - audience: Short description of the target audience (age group, style, interest)
        - setting: Description of the setting or environment
        - emotions: List of 2–4 emotional tones conveyed by the visuals, music, or pace
        - visible_text: Any on-screen text, slogans, or UI elements
        - brand: Brand name shown or implied
        - music: Brief description of background music or sound
        - call_to_action: What viewers are encouraged to do
        - keywords: 5–7 search-style keyword phrases
        - category: Short campaign category label (e.g., "tech lifestyle", "healthcare promo")

        Follow this example format **exactly** (no markdown, no quotes around field names, and clean plain text values only):
        Example format:

        {
          "product": "NeonGlow Sneakers",
          "audience": "Young adults, urban lifestyle",
          "setting": "Neon-lit city streets at night",
          "emotions": ["energized", "futuristic", "aspirational"],
          "visible_text": ["Glow Forward", "Your Light, Your Pace"],
          "brand": "ZenoFit",
          "music": "Fast-paced electronic soundtrack",
          "call_to_action": "Shop the collection now",
          "keywords": [
            "urban fashion ad",
            "tech sneakers",
            "glow in the dark shoes",
            "young professionals",
            "futuristic city",
            "fitness gear",
            "streetwear campaign"
          ],
          "category": "tech lifestyle"
        }

        DO NOT include text before or after the JSON. Output only a clean JSON object in the format above.
        """

    elif detail_level == "high":
        return """
        You are analyzing a short marketing video to extract structured metadata for search, tagging, and product-level indexing. Your output must be a clean, valid JSON object with the fields defined below. DO NOT include any markdown (no backticks, asterisks, or extra text), and DO NOT wrap your output in a string.

        ---

        ### Your output must include these fields:

        1. captions_summary  
        → A 2–3 sentence summary of what is said, shown, or narrated (spoken or visible text) throughout the video. Focus on the marketing message.

        2. description  
        → A detailed paragraph summarizing the **visual content**, atmosphere, mood, and inferred purpose of the video. Mention setting, tone, product appearance, and intent.

        3. captions  
        → Dictionary of phrases or narration keyed by timestamps (e.g., "0:03"). Include spoken lines, visible slogans, or inferred narration. Keep entries short, natural, and ad-like.

        4. keywords  
        → List of 10–15 lowercase search terms that someone might use to find this content (no punctuation or nested values).

        5. media_type  
        → Always return "video".

        6. processing_time  
        → Set to 0.

        7. timestamps  
        → Dictionary mapping timestamps to what happens visually at that moment. Use simple 1-sentence scene descriptions. Do not embed JSON inside values.

        ---

        ### Add these marketing-specific fields:

        8. product_name  
        → Name of the product or service being advertised. Use a placeholder if unknown.

        9. brand  
        → Brand or company name shown or implied.

        10. department  
        → The business function or department responsible for or relevant to the ad (e.g., "Marketing", "Retail", "Customer Experience", "Product").

        11. product_features  
        → A list of 3–5 key features or benefits shown in the video (e.g., "HD streaming", "portable design", "smart skipping").

        12. unique_selling_points  
        → A list of 2–4 bullet points summarizing why someone would choose this product over others.

        13. call_to_action  
        → What the video suggests the viewer should do next (e.g., "Buy now", "Download the app", "Stream your favorites").

        ---

        Example output:

        {
          "captions_summary": "The ad promotes watching premium TV content on the go. Phrases emphasize high-definition quality, smart video skipping, and convenience of use.",
          "description": "The 15-second ad shows a tablet streaming Game of Thrones on a wooden table beside a mug of coffee. The camera uses a static high-angle shot. The environment is calm, softly lit, and minimalist. A message appears near the end reading 'Shortened sequences'. The video suggests a seamless, relaxed entertainment experience.",
          "captions": {
            "0:02": "Stream your favorite moments anywhere.",
            "0:10": "Shortened sequences.",
            "0:14": "Available now on HBO GO."
          },
          "keywords": [
            "tablet",
            "video streaming",
            "game of thrones",
            "hd playback",
            "portable entertainment",
            "coffee table ad",
            "media device",
            "tech ad",
            "hbo go",
            "product commercial",
            "calm lifestyle",
            "brand video",
            "tv app",
            "streaming promo",
            "relaxed viewing"
          ],
          "media_type": "video",
          "processing_time": 0,
          "timestamps": {
            "0:00": "Tablet streaming Game of Thrones on a wooden table next to a coffee mug.",
            "0:10": "Text 'Shortened sequences' appears on screen.",
            "0:15": "Video ends on static product shot with branding."
          },
          "product_name": "Nexus StreamPad 7",
          "brand": "HBO GO",
          "department": "Marketing",
          "product_features": [
            "hd streaming quality",
            "smart sequence shortening",
            "portable design",
            "user-friendly interface"
          ],
          "unique_selling_points": [
            "Watch premium shows on the go",
            "High-definition playback without lag",
            "Streamlined viewing experience",
            "Optimized for mobile users"
          ],
          "call_to_action": "Download the HBO GO app and start streaming today"
        }

        DO NOT include any extra text, markdown, formatting artifacts, or headings before or after this JSON.
        """
    else:
        raise ValueError("detail_level must be 'low', 'medium', or 'high'")


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

        # Define a custom prompt
        custom_prompt = """
        Please also include the following additional fields in your analysis:

        1. target_audience_age_range → An estimated age range for the target audience (e.g., "18-34", "25-45")
        2. video_mood → Overall mood or feeling of the video (e.g., "energetic", "calm", "inspirational")
        3. marketing_funnel_stage → Where in the marketing funnel this video belongs (e.g., "awareness", "consideration", "decision")
        4. languages → List of languages used in the video
        5. competitor_analysis → Brief comparison with similar products if apparent
        """

        # Analyze the video with high detail level and custom prompt
        result = analyze_video_with_gemini(
            file_path=str(downloaded_video_path),
            api_key=api_key,
            detail_level="high",     # can also be medium or low
            custom_prompt=custom_prompt  # Add the custom prompt
        )

        # Save the analysis results to a JSON file
        result_path = output_dir / f"{os.path.splitext(video_filename)[0]}_analysis.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Analysis complete! Results saved to {result_path}")

        # Print some information about the results
        print("\nVideo Analysis Summary:")
        print(f"- Processing time: {result['processing_time']:.2f} seconds")
        print(f"- Keywords: {', '.join(result['keywords'])}")
        if 'duration' in result:
            print(f"- Duration: {result['duration']:.2f} seconds")
        print(f"\nDescription: {result['description'][:300]}...\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()