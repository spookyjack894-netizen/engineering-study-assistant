from __future__ import annotations

import base64
import io
from datetime import date, datetime
from typing import Any

import pymupdf
import streamlit as st
from groq import Groq
from openai import OpenAI
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EngineerOS",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================

TEXT_MODEL = "llama-3.3-70b-versatile"

# This model name can be changed in the sidebar if Groq updates
# its available vision models.
VISION_MODEL = "qwen/qwen3.6-27b"

MAX_TEXT_CHARACTERS = 45_000
MAX_VISION_PAGES = 3
PDF_RENDER_DPI = 150

STUDY_ACTIONS = [
    "Expand my notes",
    "Build a study guide",
    "Find missing concepts",
    "Create practice questions",
    "Explain with intuition",
    "Create flashcards",
]


# =========================================================
# SESSION STATE
# =========================================================

def initialize_session_state() -> None:
    """Create temporary app storage for the current browser session."""

    defaults: dict[str, Any] = {
        "notes_result": "",
        "vision_transcription": "",
        "tutor_result": "",
        "planner_result": "",
        "reflection_result": "",
        "interview_result": "",
        "task_list": [],
        "reflection_history": [],
        "interview_history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =========================================================
# API KEY
# =========================================================

def get_groq_api_key() -> str:
    """Read the Groq key from Streamlit Secrets."""

    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


GROQ_API_KEY = get_groq_api_key()


# =========================================================
# GENERAL AI FUNCTIONS
# =========================================================

def get_groq_client() -> Groq:
    """Create the normal Groq SDK client."""

    if not GROQ_API_KEY:
        raise ValueError(
            "Your Groq API key has not been added to Streamlit Secrets."
        )

    return Groq(api_key=GROQ_API_KEY)


def get_groq_responses_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointed toward Groq.

    We use this client for the Responses API and image inputs.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "Your Groq API key has not been added to Streamlit Secrets."
        )

    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def call_text_ai(
    system_message: str,
    user_message: str,
    model_name: str,
    temperature: float = 0.3,
    max_tokens: int = 4_000,
) -> str:
    """Send a normal text request to Groq."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise ValueError("The AI returned no response choices.")

    result = response.choices[0].message.content

    if not result:
        raise ValueError("The AI returned an empty response.")

    return result.strip()


# =========================================================
# FILE AND IMAGE FUNCTIONS
# =========================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract embedded machine-readable text from a PDF.

    This works for typed PDFs and digital lecture slides.
    Image-based and handwritten pages may return no text.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    extracted_pages: list[str] = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if page_text:
                extracted_pages.append(
                    f"\n\n--- PAGE {page_number} ---\n\n{page_text}"
                )

    finally:
        document.close()

    return "".join(extracted_pages).strip()


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF."""

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        return len(document)
    finally:
        document.close()


def render_pdf_pages(
    pdf_bytes: bytes,
    start_page: int,
    end_page: int,
    dpi: int = PDF_RENDER_DPI,
) -> list[tuple[int, bytes]]:
    """
    Render selected PDF pages as PNG images.

    Page numbers supplied by the user are 1-based.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    rendered_pages: list[tuple[int, bytes]] = []

    try:
        for page_index in range(start_page - 1, end_page):
            page = document[page_index]

            pixmap = page.get_pixmap(
                dpi=dpi,
                alpha=False,
            )

            png_bytes = pixmap.tobytes("png")

            rendered_pages.append(
                (
                    page_index + 1,
                    png_bytes,
                )
            )

    finally:
        document.close()

    return rendered_pages


def normalize_image_bytes(
    image_bytes: bytes,
    max_width: int = 1800,
) -> bytes:
    """
    Convert an uploaded image into a reasonably sized RGB JPEG.

    Reducing very large images helps avoid oversized API requests.
    """

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    if image.width > max_width:
        scale = max_width / image.width

        new_height = int(image.height * scale)

        image = image.resize(
            (max_width, new_height)
        )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    return output.getvalue()


def image_bytes_to_data_url(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """Convert image bytes into a Base64 data URL."""

    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def analyze_page_images(
    page_images: list[tuple[str, bytes]],
    vision_model: str,
    extra_context: str,
) -> str:
    """
    Ask a vision model to read handwritten or image-based notes.

    Each tuple contains:
        (page label, image bytes)
    """

    if not page_images:
        raise ValueError("No page images were supplied.")

    if len(page_images) > MAX_VISION_PAGES:
        raise ValueError(
            f"Only {MAX_VISION_PAGES} pages can be analyzed at once."
        )

    client = get_groq_responses_client()

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"""
You are examining engineering study notes.

Read the attached pages carefully.

The pages may contain:

- handwriting,
- equations,
- subscripts and superscripts,
- diagrams,
- arrows,
- graphs,
- annotations,
- chemistry notation,
- physics notation,
- partially completed problems.

Student context:

{extra_context or "No additional context was supplied."}

Create a careful page-by-page transcription and interpretation.

For each page:

1. Label the page clearly.
2. Transcribe readable writing.
3. Preserve equations using clear mathematical notation.
4. Describe diagrams, graphs, arrows, and layouts.
5. Mark uncertain readings with [uncertain].
6. Do not silently guess illegible words.
7. Identify likely subject areas and main concepts.
8. Point out anything that may have been copied incorrectly.
9. Do not solve the entire assignment unless needed to explain the page.

Finish with:

- Main concepts found
- Equations detected
- Areas needing manual verification
- Important missing explanations

Use clear Markdown.
""",
        }
    ]

    for page_label, raw_image_bytes in page_images:
        normalized_image = normalize_image_bytes(raw_image_bytes)

        content.append(
            {
                "type": "input_text",
                "text": f"Image label: {page_label}",
            }
        )

        content.append(
            {
                "type": "input_image",
                "detail": "high",
                "image_url": image_bytes_to_data_url(
                    normalized_image,
                    mime_type="image/jpeg",
                ),
            }
        )

    response = client.responses.create(
        model=vision_model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    result = response.output_text

    if not result:
        raise ValueError(
            "The vision model returned an empty response."
        )

    return result.strip()


# =========================================================
# NOTES LAB PROMPTS
# =========================================================

def build_notes_instruction(selected_action: str) -> str:
    """Return instructions for the selected Notes Lab operation."""

    instructions = {
        "Expand my notes": """
Expand incomplete explanations while preserving the student's
original organization.

Include:

1. What the notes already explain
2. Expanded explanations
3. Missing definitions
4. Equations and meanings of variables
5. Units when relevant
6. Assumptions
7. Physical or molecular intuition
8. Engineering applications
9. Common mistakes
10. Questions the student should investigate

Clearly distinguish original note content from added explanations.
""",
        "Build a study guide": """
Convert the notes into a structured engineering study guide.

Include:

1. Learning objectives
2. Main concepts
3. Essential definitions
4. Important equations
5. Variable meanings and units
6. Assumptions and limitations
7. Physical or molecular intuition
8. Conceptual examples
9. Common misconceptions
10. A final review checklist
""",
        "Find missing concepts": """
Identify ideas that appear incomplete, unexplained, disconnected,
or likely to create confusion.

For every missing area:

1. Identify the topic
2. Explain what is missing
3. Supply a careful explanation
4. Explain why it matters
5. Give one active-recall question

Do not claim that added information originally appeared in the notes.
""",
        "Create practice questions": """
Create an engineering practice set based on the notes.

Include:

1. Five conceptual questions
2. Five calculation or application questions when appropriate
3. Three intuition questions
4. Two misconception-detection questions
5. One challenging transfer problem

Place solutions in a separate answer-key section at the end.
""",
        "Explain with intuition": """
Explain the hardest concepts through:

1. Everyday intuition
2. Physical or molecular interpretation
3. Mathematical interpretation
4. Engineering application
5. Limiting cases
6. Important assumptions
7. Failure conditions

Explain what happens when important variables increase or decrease.
""",
        "Create flashcards": """
Create 15 to 25 active-recall flashcards.

Use this format:

Question:
Answer:

Include:

- definitions,
- equation meanings,
- assumptions,
- conceptual relationships,
- common mistakes,
- physical intuition,
- engineering applications.

Avoid answers that are excessively long.
""",
    }

    return instructions[selected_action]


def create_study_resource(
    notes_text: str,
    selected_action: str,
    student_context: str,
    text_model: str,
) -> str:
    """Generate a study resource from extracted or vision-read notes."""

    system_message = """
You are an engineering learning coach specializing in physics,
chemistry, mathematics, and chemical engineering.

Improve the student's reasoning, intuition, retention, and technical
communication.

Do not fabricate sources, course requirements, experiments, or equations.

Do not pretend AI-generated material originally appeared in the notes.

Use clear Markdown headings.

Flag uncertain technical information and remind the student to verify
important equations and calculations.
"""

    action_instruction = build_notes_instruction(selected_action)

    user_message = f"""
STUDENT CONTEXT

{student_context or "No additional context was provided."}


SELECTED TASK

{selected_action}


TASK INSTRUCTIONS

{action_instruction}


NOTES OR VISION TRANSCRIPTION

{notes_text}
"""

    return call_text_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=text_model,
        temperature=0.25,
        max_tokens=5_000,
    )


# =========================================================
# INTUITION TUTOR
# =========================================================

def generate_tutor_response(
    subject: str,
    topic: str,
    student_attempt: str,
    tutor_mode: str,
    text_model: str,
) -> str:
    """Generate a Socratic or explanatory tutor response."""

    system_message = """
You are a Socratic engineering tutor.

Help the student develop independent reasoning instead of immediately
giving an answer.

When an attempt is provided:

1. Identify what is correct
2. Identify the first important error
3. Explain why it is an error
4. Give the smallest useful hint
5. Ask one question that moves the student forward

Explain the physical meaning of equations.
"""

    mode_instructions = {
        "Socratic hints": """
Do not solve the entire problem immediately.
Give one hint and one forward-moving question.
""",
        "Intuition builder": """
Explain the topic through physical, molecular, mathematical,
and engineering perspectives.
""",
        "Equation explorer": """
Explain where the equation comes from, every variable, units,
assumptions, limiting cases, and failure conditions.
""",
        "Misconception check": """
Identify hidden misconceptions and give a contrasting example.
""",
        "Full explanation": """
Give a complete teaching explanation with reasoning, equations,
assumptions, and a self-check question.
""",
    }

    user_message = f"""
SUBJECT

{subject}


TOPIC OR PROBLEM

{topic}


STUDENT ATTEMPT

{student_attempt or "No attempt has been provided."}


MODE

{tutor_mode}


MODE INSTRUCTION

{mode_instructions[tutor_mode]}
"""

    return call_text_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=text_model,
        temperature=0.3,
        max_tokens=3_000,
    )


# =========================================================
# PLANNER
# =========================================================

def create_schedule_analysis(
    tasks: list[dict[str, Any]],
    available_hours: float,
    energy_level: int,
    planning_notes: str,
    text_model: str,
) -> str:
    """Generate a realistic daily study plan."""

    task_sections: list[str] = []

    for index, task in enumerate(tasks, start=1):
        task_sections.append(
            f"""
Task {index}
Name: {task["name"]}
Subject: {task["subject"]}
Deadline: {task["deadline"]}
Estimated minutes: {task["minutes"]}
Difficulty: {task["difficulty"]}/5
Importance: {task["importance"]}/5
"""
        )

    system_message = """
You are a workload analyst for an engineering student.

Create realistic schedules rather than overly optimistic schedules.

Use deadlines, importance, difficulty, available time, and cognitive
energy.

Include breaks and transition time.

Never schedule more work than fits.
"""

    user_message = f"""
AVAILABLE STUDY HOURS

{available_hours}


ENERGY LEVEL

{energy_level}/10


TASKS

{"".join(task_sections)}


OTHER CONTEXT

{planning_notes or "No additional context was supplied."}


CREATE:

1. Priority order
2. Recommended study blocks
3. Break schedule
4. What should not be attempted today
5. Main risk
6. Simpler fallback plan
"""

    return call_text_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=text_model,
        temperature=0.2,
        max_tokens=2_500,
    )


# =========================================================
# REFLECTION
# =========================================================

def analyze_reflection(
    planned_work: str,
    completed_work: str,
    time_surprise: str,
    unclear_concept: str,
    distraction: str,
    energy_level: int,
    tomorrow_change: str,
    text_model: str,
) -> str:
    """Analyze an end-of-day reflection."""

    system_message = """
You are an engineering student's reflection coach.

Identify useful patterns without being judgmental.

Focus on planning accuracy, time estimation, energy, distractions,
knowledge gaps, and one or two realistic improvements.
"""

    user_message = f"""
DATE

{date.today().isoformat()}


PLANNED WORK

{planned_work}


COMPLETED WORK

{completed_work}


TIME SURPRISE

{time_surprise}


UNCLEAR CONCEPT

{unclear_concept}


MAIN DISTRACTION

{distraction}


ENERGY

{energy_level}/10


PROPOSED CHANGE FOR TOMORROW

{tomorrow_change}


PROVIDE:

1. What went well
2. Main planning lesson
3. Main learning gap
4. One adjustment for tomorrow
5. One active-recall question
"""

    return call_text_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=text_model,
        temperature=0.25,
        max_tokens=2_000,
    )


# =========================================================
# INTERVIEW GYM
# =========================================================

def create_interview_feedback(
    interview_type: str,
    target_role: str,
    question: str,
    answer: str,
    text_model: str,
) -> str:
    """Evaluate one written interview response."""

    system_message = """
You are an engineering internship interview coach.

Give candid and constructive feedback.

For behavioral responses, evaluate STAR structure, evidence, ownership,
results, reflection, and conciseness.

For technical responses, evaluate problem framing, assumptions,
accuracy, safety awareness, communication, and uncertainty.

Never invent an experience for the student.
"""

    user_message = f"""
INTERVIEW TYPE

{interview_type}


TARGET ROLE

{target_role or "General engineering internship"}


QUESTION

{question}


STUDENT ANSWER

{answer}


PROVIDE:

1. Score from 1 to 10
2. What was strong
3. What was unclear
4. Missing evidence or reasoning
5. Stronger answer structure
6. Revised example using only supplied facts
7. One likely follow-up question
"""

    return call_text_ai(
        system_message=system_message,
        user_message=user_message,
        model_name=text_model,
        temperature=0.25,
        max_tokens=2_500,
    )


# =========================================================
# ERROR HANDLING
# =========================================================

def display_ai_error(error: Exception) -> None:
    """Display a readable error without exposing a full traceback."""

    error_text = str(error)
    lower_error = error_text.lower()

    st.error("The AI request could not be completed.")

    if "api key" in lower_error or "authentication" in lower_error:
        st.warning(
            "Check the GROQ_API_KEY entry in Streamlit Secrets."
        )

    elif "model" in lower_error and (
        "not found" in lower_error
        or "does not exist" in lower_error
        or "decommissioned" in lower_error
    ):
        st.warning(
            "That model may not be enabled or may have changed. "
            "Try another model name in the sidebar."
        )

    elif "rate limit" in lower_error or "429" in lower_error:
        st.warning(
            "You may have reached a free usage limit. Wait briefly "
            "and try fewer pages."
        )

    elif "too large" in lower_error or "context" in lower_error:
        st.warning(
            "The request may be too large. Try one to three pages."
        )

    else:
        st.warning(
            "Open Manage app → Logs for more technical information."
        )

    with st.expander("Technical error details"):
        st.code(error_text)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("🧪 EngineerOS")

    st.caption(
        "Your personal engineering learning system"
    )

    st.markdown("---")

    text_model = st.text_input(
        "Text model",
        value=TEXT_MODEL,
        help="Used for study guides, planning, tutoring, and feedback.",
    )

    vision_model = st.text_input(
        "Vision model",
        value=VISION_MODEL,
        help="Used to read scanned and handwritten pages.",
    )

    if GROQ_API_KEY:
        st.success("Groq API key connected")
    else:
        st.error("Groq API key missing")

    st.markdown("---")

    st.subheader("Learning principle")

    st.write(
        """
Use AI to strengthen your thinking—not replace it.

Review every visual transcription because handwriting, equations,
and diagrams can be misread.
"""
    )

    st.markdown("---")
    st.caption("EngineerOS v3 · Vision Notes")


# =========================================================
# HEADER
# =========================================================

st.title("🧠 EngineerOS")

st.write(
    """
A personal AI workspace for engineering notes, intuition,
planning, reflection, and interview preparation.
"""
)

if not GROQ_API_KEY:
    st.warning(
        """
Add your Groq API key under:

**Manage app → Settings → Secrets**
"""
    )


# =========================================================
# MAIN TABS
# =========================================================

notes_tab, tutor_tab, planner_tab, reflection_tab, interview_tab = st.tabs(
    [
        "📚 Vision Notes Lab",
        "🧠 Intuition Tutor",
        "📅 Daily Planner",
        "📝 Reflection",
        "💼 Interview Gym",
    ]
)


# =========================================================
# VISION NOTES LAB
# =========================================================

with notes_tab:
    st.header("📚 Vision Notes Lab")

    st.write(
        """
Upload a typed PDF, scanned PDF, handwritten PDF, image,
or camera photograph.
"""
    )

    input_method = st.radio(
        "Choose an input method",
        [
            "Upload PDF",
            "Upload image",
            "Use camera",
        ],
        horizontal=True,
    )

    selected_action = st.selectbox(
        "Study action",
        STUDY_ACTIONS,
    )

    student_context = st.text_area(
        "Optional context",
        placeholder=(
            "Example: These are Physics II notes. Please pay close "
            "attention to vectors, signs, and handwritten equations."
        ),
        height=100,
    )

    source_text = ""
    vision_images: list[tuple[str, bytes]] = []
    ready_for_vision = False

    if input_method == "Upload PDF":
        uploaded_pdf = st.file_uploader(
            "Upload one PDF",
            type=["pdf"],
            key="vision_pdf_uploader",
        )

        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.getvalue()

            try:
                page_count = get_pdf_page_count(pdf_bytes)
                source_text = extract_pdf_text(pdf_bytes)

                st.info(
                    f"PDF contains {page_count} page(s)."
                )

                if source_text:
                    st.success(
                        f"Embedded text was found: "
                        f"{len(source_text):,} characters."
                    )

                    with st.expander("Preview embedded text"):
                        st.text_area(
                            "Extracted PDF text",
                            value=source_text[:15_000],
                            height=300,
                            disabled=True,
                        )

                    processing_choice = st.radio(
                        "How should this PDF be processed?",
                        [
                            "Use embedded text",
                            "Use vision on selected pages",
                        ],
                        help=(
                            "Choose vision when the PDF contains "
                            "handwriting, diagrams, or badly extracted text."
                        ),
                    )

                else:
                    st.warning(
                        "No embedded text was found. Vision mode is required."
                    )

                    processing_choice = "Use vision on selected pages"

                if processing_choice == "Use vision on selected pages":
                    page_col1, page_col2 = st.columns(2)

                    with page_col1:
                        start_page = st.number_input(
                            "Starting page",
                            min_value=1,
                            max_value=page_count,
                            value=1,
                            step=1,
                        )

                    maximum_end_page = min(
                        page_count,
                        int(start_page) + MAX_VISION_PAGES - 1,
                    )

                    with page_col2:
                        end_page = st.number_input(
                            "Ending page",
                            min_value=int(start_page),
                            max_value=maximum_end_page,
                            value=maximum_end_page,
                            step=1,
                        )

                    selected_count = int(end_page - start_page + 1)

                    st.caption(
                        f"{selected_count} page(s) selected. "
                        f"Maximum per request: {MAX_VISION_PAGES}."
                    )

                    if st.button(
                        "Read selected pages with vision",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner(
                            "Rendering and reading the selected pages..."
                        ):
                            try:
                                rendered_pages = render_pdf_pages(
                                    pdf_bytes=pdf_bytes,
                                    start_page=int(start_page),
                                    end_page=int(end_page),
                                )

                                vision_images = [
                                    (
                                        f"PDF page {page_number}",
                                        image_bytes,
                                    )
                                    for page_number, image_bytes
                                    in rendered_pages
                                ]

                                st.session_state[
                                    "vision_transcription"
                                ] = analyze_page_images(
                                    page_images=vision_images,
                                    vision_model=vision_model,
                                    extra_context=student_context,
                                )

                            except Exception as error:
                                display_ai_error(error)

                    if st.session_state["vision_transcription"]:
                        st.subheader("Vision transcription")

                        st.markdown(
                            st.session_state[
                                "vision_transcription"
                            ]
                        )

                        verified = st.checkbox(
                            "I reviewed the transcription against the original pages."
                        )

                        if verified:
                            source_text = st.session_state[
                                "vision_transcription"
                            ]
                            ready_for_vision = True

                else:
                    ready_for_vision = bool(source_text)

            except Exception as error:
                st.error("The PDF could not be processed.")

                with st.expander("Technical details"):
                    st.code(str(error))

    elif input_method == "Upload image":
        uploaded_images = st.file_uploader(
            "Upload JPG, JPEG, or PNG notes",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="vision_image_uploader",
        )

        if uploaded_images:
            limited_images = uploaded_images[:MAX_VISION_PAGES]

            if len(uploaded_images) > MAX_VISION_PAGES:
                st.warning(
                    f"Only the first {MAX_VISION_PAGES} images "
                    f"will be analyzed."
                )

            preview_columns = st.columns(
                min(len(limited_images), 3)
            )

            for index, uploaded_image in enumerate(limited_images):
                with preview_columns[index % len(preview_columns)]:
                    st.image(
                        uploaded_image,
                        caption=uploaded_image.name,
                    )

            if st.button(
                "Read uploaded images with vision",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Reading the uploaded notes..."):
                    try:
                        image_items = [
                            (
                                uploaded_image.name,
                                uploaded_image.getvalue(),
                            )
                            for uploaded_image in limited_images
                        ]

                        st.session_state[
                            "vision_transcription"
                        ] = analyze_page_images(
                            page_images=image_items,
                            vision_model=vision_model,
                            extra_context=student_context,
                        )

                    except Exception as error:
                        display_ai_error(error)

            if st.session_state["vision_transcription"]:
                st.subheader("Vision transcription")

                st.markdown(
                    st.session_state[
                        "vision_transcription"
                    ]
                )

                verified = st.checkbox(
                    "I reviewed the transcription against my images."
                )

                if verified:
                    source_text = st.session_state[
                        "vision_transcription"
                    ]
                    ready_for_vision = True

    else:
        camera_image = st.camera_input(
            "Take a clear photograph of one page"
        )

        if camera_image is not None:
            st.image(
                camera_image,
                caption="Camera image",
            )

            if st.button(
                "Read camera image with vision",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Reading the photographed notes..."):
                    try:
                        st.session_state[
                            "vision_transcription"
                        ] = analyze_page_images(
                            page_images=[
                                (
                                    "Camera photograph",
                                    camera_image.getvalue(),
                                )
                            ],
                            vision_model=vision_model,
                            extra_context=student_context,
                        )

                    except Exception as error:
                        display_ai_error(error)

            if st.session_state["vision_transcription"]:
                st.subheader("Vision transcription")

                st.markdown(
                    st.session_state[
                        "vision_transcription"
                    ]
                )

                verified = st.checkbox(
                    "I reviewed the transcription against the photograph."
                )

                if verified:
                    source_text = st.session_state[
                        "vision_transcription"
                    ]
                    ready_for_vision = True

    if source_text and (
        ready_for_vision
        or input_method == "Upload PDF"
    ):
        st.markdown("---")
        st.subheader("Create study resource")

        if len(source_text) > MAX_TEXT_CHARACTERS:
            st.info(
                f"The study resource will use the first "
                f"{MAX_TEXT_CHARACTERS:,} characters."
            )

        notes_for_ai = source_text[:MAX_TEXT_CHARACTERS]

        if st.button(
            f"Create: {selected_action}",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Creating your study resource..."):
                try:
                    st.session_state[
                        "notes_result"
                    ] = create_study_resource(
                        notes_text=notes_for_ai,
                        selected_action=selected_action,
                        student_context=student_context,
                        text_model=text_model,
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["notes_result"]:
        st.markdown("---")
        st.subheader("Study Resource")

        st.markdown(
            st.session_state["notes_result"]
        )

        st.download_button(
            "Download study resource",
            data=st.session_state["notes_result"],
            file_name="engineering_study_resource.md",
            mime="text/markdown",
            use_container_width=True,
        )


# =========================================================
# INTUITION TUTOR TAB
# =========================================================

with tutor_tab:
    st.header("🧠 Intuition Tutor")

    tutor_col1, tutor_col2 = st.columns(2)

    with tutor_col1:
        subject = st.selectbox(
            "Subject",
            [
                "Physics",
                "General Chemistry",
                "Organic Chemistry",
                "Thermodynamics",
                "Fluid Mechanics",
                "Heat Transfer",
                "Mass Transfer",
                "Reaction Engineering",
                "Mathematics",
                "Other",
            ],
        )

    with tutor_col2:
        tutor_mode = st.selectbox(
            "Tutor mode",
            [
                "Socratic hints",
                "Intuition builder",
                "Equation explorer",
                "Misconception check",
                "Full explanation",
            ],
        )

    tutor_topic = st.text_area(
        "Concept, equation, or problem",
        height=140,
    )

    student_attempt = st.text_area(
        "Your attempt or current understanding",
        height=140,
    )

    if st.button(
        "Ask the tutor",
        type="primary",
        use_container_width=True,
    ):
        if not tutor_topic.strip():
            st.warning(
                "Enter a concept, equation, or problem first."
            )

        else:
            with st.spinner("Thinking through your problem..."):
                try:
                    st.session_state[
                        "tutor_result"
                    ] = generate_tutor_response(
                        subject=subject,
                        topic=tutor_topic,
                        student_attempt=student_attempt,
                        tutor_mode=tutor_mode,
                        text_model=text_model,
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["tutor_result"]:
        st.markdown("---")
        st.markdown(
            st.session_state["tutor_result"]
        )


# =========================================================
# DAILY PLANNER TAB
# =========================================================

with planner_tab:
    st.header("📅 Daily Planner")

    st.info(
        """
Google Calendar connection will be added after Vision Notes
is confirmed working. This version uses tasks entered manually.
"""
    )

    planner_col1, planner_col2 = st.columns(2)

    with planner_col1:
        available_hours = st.number_input(
            "Available study hours",
            min_value=0.5,
            max_value=16.0,
            value=4.0,
            step=0.5,
        )

    with planner_col2:
        planning_energy = st.slider(
            "Energy level",
            1,
            10,
            7,
        )

    task_col1, task_col2 = st.columns(2)

    with task_col1:
        task_name = st.text_input(
            "Task name"
        )

        task_subject = st.text_input(
            "Subject"
        )

        task_deadline = st.date_input(
            "Deadline",
            value=date.today(),
        )

    with task_col2:
        task_minutes = st.number_input(
            "Estimated minutes",
            min_value=10,
            max_value=600,
            value=60,
            step=10,
        )

        task_difficulty = st.slider(
            "Difficulty",
            1,
            5,
            3,
        )

        task_importance = st.slider(
            "Importance",
            1,
            5,
            4,
        )

    if st.button("Add task"):
        if not task_name.strip():
            st.warning("Enter a task name.")

        else:
            st.session_state["task_list"].append(
                {
                    "name": task_name.strip(),
                    "subject": task_subject.strip() or "Unspecified",
                    "deadline": task_deadline.isoformat(),
                    "minutes": int(task_minutes),
                    "difficulty": int(task_difficulty),
                    "importance": int(task_importance),
                }
            )

            st.success("Task added.")

    if st.session_state["task_list"]:
        st.subheader("Task list")

        for task_index, task in enumerate(
            st.session_state["task_list"]
        ):
            display_col, delete_col = st.columns([9, 1])

            with display_col:
                st.write(
                    f"**{task['name']}** — {task['subject']} · "
                    f"{task['minutes']} min · "
                    f"Difficulty {task['difficulty']}/5 · "
                    f"Importance {task['importance']}/5 · "
                    f"Due {task['deadline']}"
                )

            with delete_col:
                if st.button(
                    "✕",
                    key=f"delete_task_{task_index}",
                ):
                    st.session_state["task_list"].pop(task_index)
                    st.rerun()

        planning_notes = st.text_area(
            "Additional planning context"
        )

        if st.button(
            "Create my study plan",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing your workload..."):
                try:
                    st.session_state[
                        "planner_result"
                    ] = create_schedule_analysis(
                        tasks=st.session_state["task_list"],
                        available_hours=available_hours,
                        energy_level=planning_energy,
                        planning_notes=planning_notes,
                        text_model=text_model,
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["planner_result"]:
        st.markdown("---")
        st.markdown(
            st.session_state["planner_result"]
        )


# =========================================================
# REFLECTION TAB
# =========================================================

with reflection_tab:
    st.header("📝 Daily Reflection")

    planned_work = st.text_area(
        "What did you plan to complete?"
    )

    completed_work = st.text_area(
        "What did you actually complete?"
    )

    time_surprise = st.text_area(
        "What took longer or shorter than expected?"
    )

    unclear_concept = st.text_area(
        "What concept remains unclear?"
    )

    reflection_col1, reflection_col2 = st.columns(2)

    with reflection_col1:
        distraction = st.text_input(
            "Main distraction"
        )

    with reflection_col2:
        reflection_energy = st.slider(
            "Energy today",
            1,
            10,
            6,
        )

    tomorrow_change = st.text_area(
        "What would you change tomorrow?"
    )

    if st.button(
        "Analyze my reflection",
        type="primary",
        use_container_width=True,
    ):
        if not planned_work.strip() and not completed_work.strip():
            st.warning(
                "Enter what you planned or completed first."
            )

        else:
            with st.spinner("Finding useful patterns..."):
                try:
                    st.session_state[
                        "reflection_result"
                    ] = analyze_reflection(
                        planned_work=planned_work,
                        completed_work=completed_work,
                        time_surprise=time_surprise,
                        unclear_concept=unclear_concept,
                        distraction=distraction,
                        energy_level=reflection_energy,
                        tomorrow_change=tomorrow_change,
                        text_model=text_model,
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["reflection_result"]:
        st.markdown("---")
        st.markdown(
            st.session_state["reflection_result"]
        )


# =========================================================
# INTERVIEW TAB
# =========================================================

with interview_tab:
    st.header("💼 Interview Gym")

    st.info(
        """
The microphone and complete multi-question interview will be
added after Vision Notes is confirmed working. This version
keeps the written practice feature.
"""
    )

    interview_col1, interview_col2 = st.columns(2)

    with interview_col1:
        interview_type = st.selectbox(
            "Interview type",
            [
                "Behavioral",
                "Technical",
                "Project explanation",
                "Safety scenario",
                "Troubleshooting scenario",
            ],
        )

    with interview_col2:
        target_role = st.text_input(
            "Target role or company"
        )

    default_questions = {
        "Behavioral": (
            "Tell me about a time you faced a difficult team challenge."
        ),
        "Technical": (
            "How would you approach sizing a heat exchanger?"
        ),
        "Project explanation": (
            "Tell me about a technical project you completed."
        ),
        "Safety scenario": (
            "What would you do if you observed an unsafe condition?"
        ),
        "Troubleshooting scenario": (
            "A process flow rate suddenly decreases. How would "
            "you investigate the problem?"
        ),
    }

    interview_question = st.text_area(
        "Interview question",
        value=default_questions[interview_type],
    )

    interview_answer = st.text_area(
        "Your answer",
        height=220,
    )

    if st.button(
        "Evaluate my answer",
        type="primary",
        use_container_width=True,
    ):
        if not interview_answer.strip():
            st.warning("Write your answer first.")

        else:
            with st.spinner("Evaluating your response..."):
                try:
                    st.session_state[
                        "interview_result"
                    ] = create_interview_feedback(
                        interview_type=interview_type,
                        target_role=target_role,
                        question=interview_question,
                        answer=interview_answer,
                        text_model=text_model,
                    )

                except Exception as error:
                    display_ai_error(error)

    if st.session_state["interview_result"]:
        st.markdown("---")
        st.markdown(
            st.session_state["interview_result"]
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    """
EngineerOS is a learning assistant, not an authoritative engineering
reference. Verify equations, calculations, handwriting transcriptions,
safety decisions, and course requirements.
"""
)
