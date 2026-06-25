import gradio as gr
import torch
import spaces
import os
import sys
import tempfile
import shutil
import ast
from PIL import Image, ImageDraw, ImageFont, ImageOps
import re
import numpy as np
import base64
from io import StringIO, BytesIO
from transformers import AutoModel, AutoTokenizer
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable

colors.dodger_blue = colors.Color(
    name="dodger_blue",
    c50="#F0F8FF",
    c100="#E6F2FF",
    c200="#B8D9FF",
    c300="#8AC0FF",
    c400="#5CA7FF",
    c500="#1E90FF",
    c600="#1A7FE6",
    c700="#166ECC",
    c800="#125DB3",
    c900="#0E4C99",
    c950="#0A3B80",
)

class DodgerBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.dodger_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

dodger_blue_theme = DodgerBlueTheme()

print("Determining device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

DEFAULT_MODEL = "DeepSeek-OCR-2"

MODEL_REGISTRY = {
    "DeepSeek-OCR-2": {
        "repo": "deepseek-ai/DeepSeek-OCR-2",
        "load_kwargs": dict(
            _attn_implementation="flash_attention_2",  # prebuilt wheels @ (strangertoolshf/flash_attention_2_wheelhouse)
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=True,
        ),
        "infer_kwargs": {},
    },
    "Unlimited-OCR": {
        "repo": "baidu/Unlimited-OCR",
        "load_kwargs": dict(
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ),
        "infer_kwargs": dict(
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
        ),
    },
}

RESOLUTION_CONFIGS = {
    "DeepSeek-OCR-2": {
        "Default": {"base_size": 1024, "image_size": 768, "crop_mode": True},
        "Quality": {"base_size": 1280, "image_size": 960, "crop_mode": True},
        "Fast": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        "No Crop": {"base_size": 1024, "image_size": 768, "crop_mode": False},
        "Small": {"base_size": 768, "image_size": 512, "crop_mode": False},
    },
    "Unlimited-OCR": {
        "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    },
}

_LOADED_MODELS = {}

def get_model(model_choice):
    """Lazy-load + cache (tokenizer, model) for the selected backbone."""
    if model_choice in _LOADED_MODELS:
        return _LOADED_MODELS[model_choice]

    cfg = MODEL_REGISTRY[model_choice]
    repo = cfg["repo"]
    print(f"Loading {model_choice} ({repo})...")

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(repo, **cfg["load_kwargs"])
    mdl = mdl.eval().cuda()

    _LOADED_MODELS[model_choice] = (tok, mdl)
    print(f"✅ {model_choice} loaded")
    return tok, mdl

# Preload the default model at startup (same eager behavior as before).
get_model(DEFAULT_MODEL)

TASK_PROMPTS = {
    "Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "OCR Image": {"prompt": "<image>\n<|grounding|>OCR this image.", "has_grounding": True},
    "Parse Figure": {"prompt": "<image>\nParse the figure.", "has_grounding": False},
    "Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "Custom": {"prompt": "", "has_grounding": False}
}

REF_DET_PATTERN = re.compile(r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>', re.DOTALL)
DET_ONLY_PATTERN = re.compile(
    r'<\|det\|>([^\[<]+?)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]<\|/det\|>',
    re.DOTALL
)

def extract_grounding_references(text):
    refs = []
    matched_spans = []

    for m in REF_DET_PATTERN.finditer(text):
        label, boxes_str = m.group(1), m.group(2)
        try:
            boxes = ast.literal_eval(boxes_str)
            if boxes and not isinstance(boxes[0], (list, tuple)):
                boxes = [boxes]
        except (ValueError, SyntaxError):
            continue
        refs.append((m.group(0), label.strip(), boxes))
        matched_spans.append((m.start(), m.end()))

    for m in DET_ONLY_PATTERN.finditer(text):
        if any(start <= m.start() < end for start, end in matched_spans):
            continue
        label, x1, y1, x2, y2 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        refs.append((m.group(0), label.strip(), [[int(x1), int(y1), int(x2), int(y2)]]))

    return refs

def get_font(size=15):
    """Attempt to load a font, falling back to default if necessary."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
        "arial.ttf", 
        "Arial.ttf"
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            continue
    return ImageFont.load_default()

def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = get_font(15)
    crops = []
    
    color_map = {}
    np.random.seed(42)

    for _, label, coords in refs:
        if label not in color_map:
            color_map[label] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))

        color = color_map[label]
        color_a = color + (60,)
        
        for box in coords:
            x1, y1, x2, y2 = int(box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)
            
            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))
            
            width = 5 if label == 'title' else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)
            
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops

def clean_output(text, include_images=False):
    if not text:
        return ""
    img_num = 0

    # DeepSeek-OCR-2 style: tag pair is its own metadata line — drop the line.
    for m in list(REF_DET_PATTERN.finditer(text)):
        full, label = m.group(0), m.group(1).strip()
        if label == 'image':
            if include_images:
                text = text.replace(full, f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(full, '', 1)
        else:
            text = re.sub(rf'(?m)^[^\n]*{re.escape(full)}[^\n]*\n?', '', text, count=1)

    # Unlimited-OCR style: tag is inline, immediately followed by the actual
    # recognized text on the same line — strip only the tag itself, never
    # the line, or the OCR'd text right after it gets deleted too.
    for m in list(DET_ONLY_PATTERN.finditer(text)):
        full, label = m.group(0), m.group(1).strip()
        if label == 'image':
            if include_images:
                text = text.replace(full, f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(full, '', 1)
        else:
            text = text.replace(full, '', 1)
    
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    
    return text.strip()

def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(f'**[Figure {i + 1}]**', f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown

@spaces.GPU
def process_image(image, model_choice, mode, task, custom_prompt):
    if image is None:
        return "Error: Upload an image", "", "", None, []
    if task in ["Custom", "Locate"] and not custom_prompt.strip():
        return "Please enter a prompt", "", "", None, []

    tokenizer, model = get_model(model_choice)

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)

    resolution_map = RESOLUTION_CONFIGS[model_choice]
    config = resolution_map.get(mode) or next(iter(resolution_map.values()))

    if task == "Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task == "Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()
    
    stdout = sys.stdout
    sys.stdout = StringIO()

    model.infer(
        tokenizer=tokenizer,
        prompt=prompt,
        image_file=tmp.name,
        output_path=out_dir,
        base_size=config["base_size"],
        image_size=config["image_size"],
        crop_mode=config["crop_mode"],
        save_results=False,
        **MODEL_REGISTRY[model_choice]["infer_kwargs"]
    )
    
    result = '\n'.join([l for l in sys.stdout.getvalue().split('\n') 
                        if not any(s in l for s in ['image:', 'other:', 'PATCHES', '====', 'BASE:', '%|', 'torch.Size'])]).strip()
    sys.stdout = stdout
    
    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    
    if not result:
        return "No text detected", "", "", None, []
    
    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)
    
    img_out = None
    crops = []

    if has_grounding:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)
    
    markdown = embed_images(markdown, crops)
    
    return cleaned, markdown, result, img_out, crops

def toggle_prompt(task):
    if task == "Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Add <|grounding|> for bounding boxes")
    elif task == "Locate":
        return gr.update(visible=True, label="Text to Locate", placeholder="Enter text to locate")
    return gr.update(visible=False)

def select_boxes(task):
    if task == "Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()

def update_resolution_choices(model_choice):
    choices = list(RESOLUTION_CONFIGS[model_choice].keys())
    return gr.update(choices=choices, value=choices[0])


css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* Background grid pattern - Dodger Blue theme */
body, .gradio-container {
    background-color: #F0F8FF !important;
    background-image: 
        linear-gradient(#B8D9FF 1px, transparent 1px), 
        linear-gradient(90deg, #B8D9FF 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Dark mode grid */
.dark body, .dark .gradio-container {
    background-color: #1a1a1a !important;
    background-image: 
        linear-gradient(rgba(30, 144, 255, 0.1) 1px, transparent 1px), 
        linear-gradient(90deg, rgba(30, 144, 255, 0.1) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
}

#col-container {
    margin: 0 auto;
    max-width: 1000px;
}

/* Main title styling */
#main-title {
    text-align: center !important;
    padding: 1rem 0 0.5rem 0;
}

#main-title h1 {
    font-size: 2.5em !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #1E90FF 0%, #5CA7FF 50%, #1A7FE6 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease infinite;
    letter-spacing: -0.02em;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Subtitle styling */
#subtitle {
    text-align: center !important;
    margin-bottom: 1.5rem;
}

#subtitle p {
    margin: 0 auto;
    color: #666666;
    font-size: 1rem;
}

#subtitle a {
    color: #1E90FF !important;
    text-decoration: none;
    font-weight: 500;
}

#subtitle a:hover {
    text-decoration: underline;
}

/* Card styling */
.gradio-group {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #B8D9FF !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 24px rgba(30, 144, 255, 0.08) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.gradio-group:hover {
    box-shadow: 0 8px 32px rgba(30, 144, 255, 0.12) !important;
    border-color: #5CA7FF !important;
}

.dark .gradio-group {
    background: rgba(30, 30, 30, 0.9) !important;
    border-color: rgba(30, 144, 255, 0.3) !important;
}

/* Image upload area */
.gradio-image {
    border-radius: 10px !important;
    overflow: hidden;
    border: 2px dashed #5CA7FF !important;
    transition: all 0.3s ease;
}

.gradio-image:hover {
    border-color: #1E90FF !important;
    background: rgba(30, 144, 255, 0.02) !important;
}

/* Radio buttons */
.gradio-radio {
    border-radius: 8px !important;
}

.gradio-radio label {
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}

.gradio-radio label:hover {
    background: rgba(30, 144, 255, 0.05) !important;
}

.gradio-radio label.selected {
    background: rgba(30, 144, 255, 0.1) !important;
    border-color: #1E90FF !important;
}

/* Primary button */
.primary {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
}

/* Tabs styling */
.tab-nav {
    border-bottom: 2px solid #B8D9FF !important;
}

.tab-nav button {
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: rgba(30, 144, 255, 0.1) !important;
    border-bottom: 2px solid #1E90FF !important;
}

/* Output textbox */
.gradio-textbox textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #B8D9FF !important;
    border-radius: 8px !important;
}

.dark .gradio-textbox textarea {
    background: rgba(30, 30, 30, 0.95) !important;
    border-color: rgba(30, 144, 255, 0.2) !important;
}

/* Markdown output */
.gradio-markdown {
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.7 !important;
}

.gradio-markdown code {
    font-family: 'IBM Plex Mono', monospace !important;
    background: rgba(30, 144, 255, 0.08) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    color: #166ECC !important;
}

.gradio-markdown pre {
    background: rgba(30, 144, 255, 0.05) !important;
    border: 1px solid #B8D9FF !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

/* Examples section */
.gradio-examples {
    border-radius: 10px !important;
}

.gradio-examples .gallery-item {
    border: 2px solid #B8D9FF !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gradio-examples .gallery-item:hover {
    border-color: #1E90FF !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(30, 144, 255, 0.15) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(30, 144, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #1E90FF, #5CA7FF);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #1A7FE6, #1E90FF);
}

/* Accordion styling */
.gradio-accordion {
    border-radius: 10px !important;
    border: 1px solid #B8D9FF !important;
}

.gradio-accordion > .label-wrap {
    background: rgba(30, 144, 255, 0.03) !important;
    border-radius: 10px !important;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-row {
    animation: fadeIn 0.4s ease-out;
}

/* Label styling */
label {
    font-weight: 600 !important;
    color: #333 !important;
}

.dark label {
    color: #eee !important;
}

/* Dropdown styling */
.gradio-dropdown {
    border-radius: 8px !important;
}

.gradio-dropdown select, .gradio-dropdown input {
    border: 1px solid #B8D9FF !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gradio-dropdown select:focus, .gradio-dropdown input:focus {
    border-color: #1E90FF !important;
    box-shadow: 0 0 0 2px rgba(30, 144, 255, 0.1) !important;
}

/* Gallery styling */
.gradio-gallery {
    border-radius: 10px !important;
}

.gradio-gallery .gallery-item {
    border: 2px solid #B8D9FF !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gradio-gallery .gallery-item:hover {
    border-color: #1E90FF !important;
    box-shadow: 0 4px 12px rgba(30, 144, 255, 0.15) !important;
}
"""

with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-OCR-2 & Unlimited-OCR", elem_id="main-title")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])

            model_choice = gr.Dropdown(
                list(MODEL_REGISTRY.keys()),
                value=DEFAULT_MODEL,
                label="Model"
            )
            mode = gr.Dropdown(
                list(RESOLUTION_CONFIGS[DEFAULT_MODEL].keys()),
                value="Default",
                label="Resolution"
            )
            task = gr.Dropdown(list(TASK_PROMPTS.keys()), value="Markdown", label="Task")
            prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
            btn = gr.Button("Perform OCR", variant="primary", size="lg")
            
            examples = gr.Examples(
                examples=["examples/1.jpg", "examples/2.jpg", "examples/3.jpg"],
                inputs=image_input, 
                label="Examples"
            )
        
        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.Tab("Text", id="tab_text"):
                    text_out = gr.Textbox(lines=20, show_label=False)
                with gr.Tab("Markdown Preview", id="tab_markdown"):
                    md_out = gr.Markdown("")
                with gr.Tab("Boxes", id="tab_boxes"):
                    img_out = gr.Image(type="pil", height=500, show_label=False)
                with gr.Tab("Cropped Images", id="tab_crops"):
                    gallery = gr.Gallery(show_label=False, columns=3, height=400)
                with gr.Tab("Raw Text", id="tab_raw"):
                    raw_out = gr.Textbox(lines=20, show_label=False)
    
            with gr.Accordion("Note", open=False):
                gr.Markdown(
                    "Inference using Huggingface transformers on NVIDIA GPUs. "
                    "Each model is lazy-loaded on first selection and then cached on GPU for the rest of the session. "
                    "Box detection on the Boxes/Cropped Images tabs works for both models — DeepSeek-OCR-2's "
                    "ref+det tag pairs and Unlimited-OCR's inline det-only tags are both parsed."
                )
    
    model_choice.change(update_resolution_choices, [model_choice], [mode])
    task.change(toggle_prompt, [task], [prompt])
    
    submit_event = btn.click(
        process_image, 
        [image_input, model_choice, mode, task, prompt],
        [text_out, md_out, raw_out, img_out, gallery]
    )
    submit_event.then(select_boxes, [task], [tabs])

if __name__ == "__main__":
    demo.queue(max_size=50).launch(theme=dodger_blue_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
