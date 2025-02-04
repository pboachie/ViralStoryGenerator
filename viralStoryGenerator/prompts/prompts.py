# viralStoryGenerator/prompts/prompts.py

def get_system_instructions():
    return (
        "You are a helpful assistant that strictly follows formatting rules. Produce exactly two sections: \n### Story Script: \n### Video Description\n\n"
        "Rules:\n"
        "1. Do NOT add extra commentary or disclaimers.\n"
        "2. Output MUST contain exactly two sections in this order:\n"
        "   - \"### Story Script:\" followed by the story\n"
        "   - \"### Video Description:\" followed by the description\n"
        "3. The video description must be a single line (<= 100 characters).\n"
        "You must produce a story script in an informal, conversational narrative style. Do not include:\n"
        "1. Scenes, stage directions, or camera shot descriptions (e.g., '[Cut to…]').\n"
        "2. Additional commentary or disclaimers.\n"
        "3. 'Let me know your thoughts,' 'like and subscribe,' or other calls to action.\n"
        "The output must only be a single narrative in paragraphs—no bullet points, no brackets.\n"
        "Output must contain exactly two sections:\n"
        "### Story Script: followed by the story.\n"
        "### Video Description: followed by the description (one line, <=100 characters).\n"


    )

def get_user_prompt(topic, sources):
    return f"""
Below are several sources and articles with notes about {topic}. Using the provided information,
please generate a short, engaging story script that is about 1.5 minutes long when narrated.
The script should be informal, conversational, and suitable for a casual video update.
Make sure to highlight the key points, include any 'spicy' or controversial details mentioned in
the notes, and explain why {topic} hasn't been working recently, while also weaving in any speculations
or rumors as appropriate.

Additionally, please generate a video description that is a maximum of 100 characters long.
The description should include creatively placed hashtags related to the subject of the story.

Here are the sources and notes:
{sources}

Now, please produce the narrated story script followed by the video description.
""".strip()

def get_fix_prompt(raw_text):
    return f"""
You provided the following text, but it doesn't follow the required format:

{raw_text}

Reformat this text to exactly include two sections:
1) ### Story Script:
2) ### Video Description:

No additional text or sections.
The video description must be a single line with a maximum of 100 characters.
"""

def get_storyboard_prompt(story):
    return f"""
You are a storyboard generator that takes a story script and produces a detailed storyboard in JSON format.
The JSON must have a key "scenes" which is a list of scenes. Each scene must include:
  - "scene_number": a sequential integer starting at 1.
  - "narration_text": the portion of the story to be narrated in this scene.
  - "image_prompt": an ultra-detailed description for generating an image using DALL·E 3.
  - "duration": estimated duration in seconds (round to the nearest integer).
    (Estimate the duration based on the length of the narration text assuming an average reading speed of 150 words per minute.)
Output only valid JSON.
Story:
{story}
"""