import os
os.environ["HUGGINGFACE_TOKEN"] = "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"
import streamlit as st
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import time

# ----------------------------------------
# Page config + CSS
# ----------------------------------------
st.set_page_config(page_title="AI Fashion Design Studio", page_icon="👗", layout="centered")

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(120deg,#fff6fb 0%, #f0fbff 100%);
        color: #222;
    }
    .block-container {
        max-width: 1000px;
        padding-top: 1.5rem;
    }
    /* Title style */
    .big-title {
        font-size:28px;
        color: #b83280;
        font-weight:700;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#b83280,#d94e9f);
        color: white;
        font-weight:600;
        border-radius:10px;
        padding: 8px 16px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    /* Sidebar */
    .stSidebar .sidebar-content {
        background: linear-gradient(180deg,#fff0f5,#fff6fb);
        border-radius:10px;
        padding:12px;
    }
    .card {
        background: white;
        padding:12px;
        border-radius:10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------
# Utility: generate fast placeholder image (no external APIs)
# ----------------------------------------
def create_placeholder_image(prompt_text, size=(512, 512), seed=None):
    """Create a stylized placeholder image showing the prompt text on a gradient background."""
    if seed is not None:
        np.random.seed(seed)
    w, h = size
    base = Image.new("RGB", (w, h), "#ffffff")
    # gradient
    arr = np.linspace(0, 1, w)
    r = (np.outer(np.ones(h), arr) * 230 + np.random.randint(0, 25)).astype(np.uint8)
    g = (np.outer(np.ones(h), arr[::-1]) * 200 + np.random.randint(0, 25)).astype(np.uint8)
    b = np.full((h, w), 240, dtype=np.uint8)
    grad = np.dstack([r, g, b])
    img = Image.fromarray(grad)
    draw = ImageDraw.Draw(img)
    # add semi-transparent rectangle
    overlay = Image.new("RGBA", img.size, (255,255,255,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(20, h-140), (w-20, h-20)], fill=(255,255,255,180))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    # write prompt text
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=22)
    except Exception:
        font = ImageFont.load_default()
    # wrap text
    lines = []
    words = prompt_text.split()
    line = ""
    for word in words:
        if len(line + " " + word) > 32:
            lines.append(line.strip())
            line = word
        else:
            line += " " + word
    if line:
        lines.append(line.strip())
    y_text = h - 130
    x_text = 40
    for ln in lines[:5]:
        draw.text((x_text, y_text), ln, fill=(30,30,30), font=font)
        y_text += 24
    # small decoration
    draw.text((20,20), "AI Demo", fill=(255,255,255), font=font)
    return img

# ----------------------------------------
# Helper: extract color palette using KMeans
# ----------------------------------------
def extract_palette(pil_img, n_colors=5):
    arr = np.array(pil_img.resize((128,128))).reshape(-1,3)
    try:
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
        centers = kmeans.cluster_centers_.astype(int)
    except Exception:
        # fallback: sample quantized colors
        centers = np.unique(arr.reshape(-1,3), axis=0)[:n_colors]
        if centers.shape[0] < n_colors:
            # pad with white
            pad = np.zeros((n_colors - centers.shape[0], 3), dtype=int) + 255
            centers = np.vstack([centers, pad])
    return centers

# ----------------------------------------
# Cached resource: optionally load Hugging Face pipeline if HF token exists
# ----------------------------------------
@st.cache_resource
def load_sd_pipeline(hf_token: str = None, model_name: str = "runwayml/stable-diffusion-v1-5"):
    """Attempt to load a Stable Diffusion pipeline if HF token provided. Return pipeline or None."""
    try:
        # lazy import for speed
        from diffusers import StableDiffusionPipeline
        import torch
        if hf_token:
            # Use auth_token for models that require it
            pipe = StableDiffusionPipeline.from_pretrained(model_name, use_auth_token=hf_token)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_name)
        # optimize dtype if cuda available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cpu")
        return pipe
    except Exception as e:
        # Failure (e.g., missing token or package). Return None.
        return None

# ----------------------------------------
# UI: Sidebar - Mode selection and keys
# ----------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/892/892458.png", width=90)
st.sidebar.title("Mode & Keys")
mode = st.sidebar.selectbox("Choose mode", ["Auto-detect (recommended)", "Demo Only (fast placeholder)", "HuggingFace SD (if token)", "OpenAI (paid)"])
st.sidebar.markdown("---")

# Provide optional secrets
hf_token_input = st.sidebar.text_input("HuggingFace Token (optional)", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
st.sidebar.markdown("**Tip:** For quick testing, leave both blank and use Demo mode.")

# Auto detect logic
use_openai = False
use_hf = False
if mode == "Auto-detect (recommended)":
    if openai_key_input:
        use_openai = True
    elif hf_token_input:
        use_hf = True
    else:
        # default to demo placeholder
        mode = "Demo Only (fast placeholder)"
        use_hf = False
        use_openai = False
elif mode == "HuggingFace SD (if token)":
    use_hf = bool(hf_token_input)
elif mode == "OpenAI (paid)":
    use_openai = bool(openai_key_input)
elif mode == "Demo Only (fast placeholder)":
    pass

# If OpenAI chosen, configure
if use_openai:
    try:
        import openai
        openai.api_key = openai_key_input or os.getenv("OPENAI_API_KEY")
    except Exception:
        use_openai = False

# If HF chosen, attempt to load pipeline
sd_pipeline = None
if use_hf:
    hf_token = hf_token_input or os.getenv("HUGGINGFACE_TOKEN")
    with st.spinner("Loading Stable Diffusion pipeline (may take ~15-45s first time)..."):
        sd_pipeline = load_sd_pipeline(hf_token)
        if sd_pipeline is None:
            st.sidebar.warning("Unable to load SD pipeline. Will fall back to placeholder demo images.")
            use_hf = False

# ----------------------------------------
# UI Pages: Navigation
# ----------------------------------------
page = st.sidebar.radio("Navigation", ["Home", "Inputs", "Generate", "Outputs", "About"])

# Shared states
if "user_input" not in st.session_state:
    st.session_state["user_input"] = {}

if page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("👗 AI Fashion Design Studio — Dual Mode")
    st.markdown(
        """
        Create fashion designs using:
        - **OpenAI** (paid) — real AI images & text (enter OpenAI key),
        - **Hugging Face Stable Diffusion** (if you provide HF token),
        - **Fast Demo placeholders** (no external API; great for demos & sharing).

        Use the **Inputs** page to enter your idea, then **Generate** to produce designs,
        and **Outputs** to preview & export PDF.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Inputs Page ----------------
elif page == "Inputs":
    st.header("✍️ Describe the Outfit / Idea")
    with st.form("input_form"):
        prompt = st.text_area("Describe your outfit idea (be descriptive):",
                              placeholder="e.g., A modern pastel lehenga with delicate floral embroidery and sheer dupatta")
        gender = st.selectbox("Target Gender", ["Unisex", "Female", "Male"])
        age_group = st.selectbox("Age Group", ["Child", "Teen", "Adult", "Senior"])
        style = st.selectbox("Style", ["Any", "Casual", "Formal", "Party", "Traditional", "Streetwear", "Ethnic", "Minimalist"])
        n_designs = st.slider("Number of designs to generate (demo may restrict to 2)", 1, 4, 3)
        allow_upload = st.file_uploader("Optional: Upload inspiration image", type=["jpg","jpeg","png"])
        submitted = st.form_submit_button("Save Inputs")
    if submitted:
        st.session_state["user_input"] = {
            "prompt": prompt.strip(),
            "gender": gender,
            "age_group": age_group,
            "style": style,
            "n_designs": n_designs,
            "uploaded": allow_upload
        }
        st.success("Inputs saved! Go to Generate page.")

# ---------------- Generate Page ----------------
elif page == "Generate":
    st.header("🚀 Generate Designs")
    ui = st.session_state.get("user_input", {})
    if not ui or not ui.get("prompt"):
        st.warning("Please provide inputs on the Inputs page first.")
    else:
        st.markdown(f"**Prompt:** {ui['prompt']}")
        # Start generation
        if st.button("Generate Now"):
            prompt_full = f"{ui['prompt']}, for a {ui['age_group'].lower()} {ui['gender'].lower()} in {ui['style'].lower()} style"
            n = min(ui.get("n_designs", 2), 4)
            results = []
            start_time = time.time()
            for i in range(n):
                st.info(f"Generating design {i+1}/{n} ...")
                # Priority order: OpenAI -> HF pipeline -> local placeholder
                img = None
                desc = ""
                if use_openai:
                    try:
                        # Text via OpenAI chat (description) + images via OpenAI images (if available)
                        # Example shown using images.generate (may be different by SDK versions)
                        import openai
                        # generate image (wrap in try)
                        try:
                            img_resp = openai.images.generate(
                                model="gpt-image-1",
                                prompt=f"Fashion design illustration of {prompt_full}, studio lighting, high quality, plain background",
                                size="512x512"
                            )
                            image_url = img_resp.data[0].url
                            image_data = requests.get(image_url).content
                            img = Image.open(BytesIO(image_data)).convert("RGB")
                        except Exception:
                            img = None
                        # description
                        try:
                            chat_resp = openai.ChatCompletion.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role":"system","content":"You are a creative fashion stylist."},
                                    {"role":"user","content":f"Describe this outfit '{prompt_full}' in 2-3 elegant lines."}
                                ],
                                max_tokens=120
                            )
                            desc = chat_resp.choices[0].message.content.strip()
                        except Exception:
                            desc = f"A stylish {ui['style'].lower()} outfit for {ui['age_group'].lower()} {ui['gender'].lower()}s."
                    except Exception as e:
                        st.warning(f"OpenAI generation failed for design {i+1}: {e}")
                        use_openai = False  # fallback next iterations

                if img is None and use_hf and sd_pipeline is not None:
                    try:
                        # generate via pipeline with optimized steps
                        generator = None
                        # if torch available, could pass generator; we'll not rely on it for simplicity
                        pipe = sd_pipeline
                        # tune steps & guidance scale for speed-quality tradeoff
                        image_out = pipe(prompt_full, num_inference_steps=20, guidance_scale=7.5).images[0]
                        img = image_out.convert("RGB")
                        desc = f"A {ui['style'].lower()} outfit inspired by '{ui['prompt']}'."
                    except Exception as e:
                        st.warning(f"HuggingFace SD generation failed for design {i+1}: {e}")
                        img = None

                if img is None:
                    # fallback placeholder (very fast)
                    img = create_placeholder_image(prompt_full, seed=i * 7)
                    desc = f"A demo-style placeholder image for: {ui['prompt'][:120]}"

                # extract palette
                palette = extract_palette(img, n_colors=5)
                results.append({"image": img, "desc": desc, "palette": palette})
                # small throttle to update UI smoothly
                time.sleep(0.6)

            elapsed = time.time() - start_time
            st.success(f"Generated {len(results)} designs in {elapsed:.1f}s")
            # store results
            st.session_state["last_results"] = results
            st.session_state["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

# ---------------- Outputs Page ----------------
elif page == "Outputs":
    st.header("🖼️ Designs & Export")
    if "last_results" not in st.session_state:
        st.info("No results to show yet — go to Generate page.")
    else:
        results = st.session_state["last_results"]
        st.markdown(f"**Generated at:** {st.session_state.get('generated_at','-')}")
        for idx, item in enumerate(results, start=1):
            with st.expander(f"Design {idx}"):
                st.image(item["image"], use_column_width=True, caption=f"Design {idx}")
                st.markdown(f"**Description:** {item['desc']}")
                # palette display
                fig, ax = plt.subplots(figsize=(5, 1))
                for j,c in enumerate(item["palette"]):
                    ax.add_patch(plt.Rectangle((j,0),1,1,color=c/255))
                ax.set_xlim(0, len(item["palette"])); ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig)
                # save local preview image for PDF
                item_path = f"fashion_design_{idx}.png"
                item["image"].save(item_path)

        # PDF export
        if st.button("📥 Export Portfolio PDF"):
            pdf_filename = f"Fashion_Portfolio_{int(time.time())}.pdf"
            pdf = SimpleDocTemplate(pdf_filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("AI Fashion Design Studio Portfolio", styles["Title"]))
            story.append(Spacer(1, 12))
            ui = st.session_state.get("user_input", {})
            story.append(Paragraph(f"<b>Prompt:</b> {ui.get('prompt','')}", styles["Normal"]))
            story.append(Paragraph(f"<b>Target:</b> {ui.get('age_group','')} {ui.get('gender','')}", styles["Normal"]))
            story.append(Paragraph(f"<b>Style:</b> {ui.get('style','')}", styles["Normal"]))
            story.append(Spacer(1, 12))
            for idx, item in enumerate(results, start=1):
                img_path = f"fashion_design_{idx}.png"
                story.append(Paragraph(f"<b>Design {idx}</b>", styles["Heading2"]))
                story.append(RLImage(img_path, width=300, height=300))
                story.append(Paragraph(item["desc"], styles["Normal"]))
                story.append(Spacer(1, 12))
            pdf.build(story)
            with open(pdf_filename, "rb") as f:
                st.download_button("Download Portfolio PDF", data=f, file_name=pdf_filename, mime="application/pdf")

# ---------------- About Page ----------------
elif page == "About":
    st.header("ℹ️ About")
    st.markdown(
        """
        **AI Fashion Design Studio** — Demo-ready app with dual modes:
        - **OpenAI mode** (paid): uses OpenAI images & chat for production-grade outputs.
        - **Hugging Face SD** (optional): when you provide HF token, will attempt local SD generation.
        - **Demo placeholders**: no external access required; instant and free to test UI.

        **Notes on speed & GPUs**
        - If using Stable Diffusion on a GPU (Colab GPU / T4 / V100 / A100), expect ~7–25s per 512×512 image depending on hardware.
        - On CPU generation is much slower (minutes per image).

        **Security**
        - Do not hardcode API keys. Use `st.secrets` or environment variables when deploying.
        """
    )

# End of file
