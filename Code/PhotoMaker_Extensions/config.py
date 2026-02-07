# config.py

# Input image(s)
INPUT_IMAGES = "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Input"

# Prompts
PROMPTS_FACE_LEFT = [
    "a man img wearing a spacesuit",
]

PROMPTS_FACE_RIGHT = [
    "a man img wearing a spacesuit",
]

# Output settings
OUTPUT_DIR = "/teamspace/studios/this_studio/PhotoMaker-CAP-C6-Group_3/Data/Output"
NUM_OUTPUTS = 1

# Style
STYLE_NAME = "Photographic (Default)"

# Negative prompt
NEGATIVE_PROMPT = (
    "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "normal quality, jpeg artifacts, signature, watermark, username, blurry"
)

# Output dimensions
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# Generation parameters
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
STYLE_STRENGTH_RATIO = 20
SEED = None

# Sketch settings
USE_SKETCH = False
SKETCH_IMAGE_PATH = None
ADAPTER_CONDITIONING_SCALE = 0.7
ADAPTER_CONDITIONING_FACTOR = 0.8
