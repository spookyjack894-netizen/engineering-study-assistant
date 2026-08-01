import io

import pymupdf
import streamlit as st
from google import genai


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Engineering Study Assistant",
    page_icon="🧪",
    layout="wide",
)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract readable text from a normal text-based PDF.
    This will not reliably read handwritten or scanned notes yet.
    """
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text("text", sort=True)

        pages.append(
            f"\n\n--- PAGE {page_number} ---\n\n"
            f"{page_text.strip()}"
        )

    document.close()
    return "".join(pages).strip()


def build_instruction(selected_action: str) -> str:
    """
    Return a different AI instruction depending on the student's goal.
    """
    instructions = {
        "Expand my notes": """
Analyze the student's notes and expand incomplete ideas.

Organize the response using:
1. Main topics
2. Expanded explanations
3. Important definitions
4. Equations and variable meanings
5. Assumptions
6. Physical, chemical, or molecular intuition
7. Connections to engineering applications
8. Common mistakes
9. Questions the student should still investigate

Clearly distinguish information found in the notes from explanations
you are adding.
""",
        "Build a study guide": """
Convert the notes into a structured engineering study guide.

Include:
1. Learning objectives
2. Major concepts
3. Key definitions
4. Important equations
5. Meaning of every variable
6. Assumptions and limitations
7. Worked conceptual examples
8. Common misconceptions
9. A concise review checklist
""",
        "Find missing concepts": """
Identify ideas that appear incomplete, unexplained, or disconnected.

For every missing area:
1. Quote or summarize the relevant part of the notes
2. Explain what appears to be missing
3. Supply a careful explanation
4. Explain why it matters
5. Give one question the student should be able to answer

Do not claim that added information originally appeared in the PDF.
""",
        "Create practice questions": """
Create an active-recall practice set based on these notes.

Include:
1. Five conceptual questions
2. Five calculation or application questions when appropriate
3. Two intuition questions
4. Two misconception-detection questions
5. One challenging transfer problem

Put all answers in a separate answer-key section at the end.
""",
        "Explain with intuition": """
Explain the most difficult concepts in the notes using four levels:

1. Everyday intuition
2. Physical or molecular interpretation
3. Mathematical interpretation
4. Engineering application

Also explain:
- what happens when major variables increase or decrease,
- which assumptions are being made,
- and when the governing equations stop working.
""",
    }

    return instructions[selected_action]


def analyze_notes(
    api_key: str,
    model_name: str,
    notes_text: str,
    selected_action: str,
    student_context: str,
) -> str:
    """
    Send the extracted notes and selected learning instruction
    to the Gemini API.
    """
    client = genai.Client(api_key=api_key)

    instruction = build_instruction(selected_action)

    prompt = f"""
You are an engineering learning coach specializing in chemistry,
physics, mathematics, and chemical engineering.

Your goal is to strengthen the student's reasoning and retention,
not simply complete assignments for them.

STUDENT CONTEXT:
{student_context or "No additional context was provided."}

SELECTED TASK:
{selected_action}

INSTRUCTIONS:
{instruction}

STUDENT NOTES:
{notes_text}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not response.text:
        raise ValueError("The AI returned an empty response.")

    return response.text


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.title("🧪 Engineering Study Assistant")

st.write(
    """
Upload class notes in PDF format, choose how you want to study them,
and receive an AI-assisted study resource.
"""
)

st.info(
    """
Start with typed lecture notes or digitally created PDFs.
Scanned handwriting may require OCR, which we will add later.
"""
)


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    model_name = st.text_input(
        "Gemini model name",
        value="gemini-2.5-flash",
        help=(
            "Use a Gemini model available to your Google AI Studio account. "
            "You can change this later."
        ),
    )

    st.markdown("---")

    st.subheader("Study principle")

    st.write(
        """
Use this tool to expand your thinking—not to replace your own work.
Always verify technical equations and calculations.
"""
    )


# ---------------------------------------------------------
# LOAD API KEY
# ---------------------------------------------------------

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    gemini_api_key = ""


# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your notes",
    type=["pdf"],
    help="Upload one PDF at a time.",
)

selected_action = st.selectbox(
    "What should the assistant do?",
    [
        "Expand my notes",
        "Build a study guide",
        "Find missing concepts",
        "Create practice questions",
        "Explain with intuition",
    ],
)

student_context = st.text_area(
    "Optional context",
    placeholder=(
        "Example: These are Physics II notes about electric fields. "
        "I understand the equations but struggle with physical intuition."
    ),
    height=100,
)


# ---------------------------------------------------------
# PROCESS THE PDF
# ---------------------------------------------------------

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()
        extracted_text = extract_pdf_text(pdf_bytes)

        if not extracted_text:
            st.error(
                "No readable text was found. This may be a scanned or "
                "handwritten PDF."
            )
        else:
            st.success(
                f"PDF processed successfully. "
                f"Approximately {len(extracted_text):,} characters were found."
            )

            with st.expander("Preview extracted notes"):
                st.text_area(
                    "Extracted PDF text",
                    value=extracted_text[:15000],
                    height=350,
                    disabled=True,
                )

            if len(extracted_text) > 50000:
                st.warning(
                    "This PDF is large. Only the first 50,000 characters "
                    "will be sent in this beginner version."
                )

            notes_for_ai = extracted_text[:50000]

            analyze_button = st.button(
                "Analyze my notes",
                type="primary",
                use_container_width=True,
            )

            if analyze_button:
                if not gemini_api_key:
                    st.error(
                        "The Gemini API key has not been added to "
                        "Streamlit Secrets yet."
                    )
                else:
                    with st.spinner("Analyzing your notes..."):
                        try:
                            result = analyze_notes(
                                api_key=gemini_api_key,
                                model_name=model_name,
                                notes_text=notes_for_ai,
                                selected_action=selected_action,
                                student_context=student_context,
                            )

                            st.session_state["latest_result"] = result

                        except Exception as error:
                            st.error(
                                "The analysis could not be completed."
                            )
                            st.exception(error)

    except Exception as error:
        st.error("The PDF could not be processed.")
        st.exception(error)


# ---------------------------------------------------------
# DISPLAY AND DOWNLOAD THE RESULT
# ---------------------------------------------------------

if "latest_result" in st.session_state:
    st.markdown("---")
    st.header("Your Study Resource")

    st.markdown(st.session_state["latest_result"])

    st.download_button(
        label="Download as Markdown",
        data=st.session_state["latest_result"],
        file_name="engineering_study_resource.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ---------------------------------------------------------
# PRIVACY NOTICE
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    """
Privacy reminder: Do not upload confidential research, private student
records, proprietary company documents, or sensitive personal information.
"""
)
