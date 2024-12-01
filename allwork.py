import streamlit as st
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
from fpdf import FPDF
import os
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain

huggingface_api_token = os.getenv("HUGGINGFACEHU_API_TOKEN")
def clean_text(text):
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text
def recording(duration, samplerate=44100):
    st.write(f"Recording for {duration} seconds...")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    st.write("Recording completed!")
    return data
# Convert audio to text
def audiototext(inputfile):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(inputfile) as source:
            st.write("Processing audio...")
            audiodata = recognizer.record(source)
            st.write("Converting audio to text...")
            text = recognizer.recognize_google(audiodata)
            return text
    except sr.UnknownValueError:
        return "Sorry, speech was unclear. Please try again."
    except sr.RequestError as e:
        return f"Could not request results from the speech recognition service; {e}"
    except Exception as ex:
        return f"An error occurred: {ex}"

# Use HuggingFace model to summarize text
def predict(text):
    llm = HuggingFaceHub(repo_id="utrobinmv/t5_summary_en_ru_zh_base_2048", model_kwargs={"temperature":0,"max_length":64})
    prompt = PromptTemplate(input_variables=['text'], template='Summarizing the following text in English: {text}')
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text)
    return summary

# Create PDF from text
def texttopdf(text, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    pdf.output(output_pdf)
    st.write(f"PDF saved as: {output_pdf}")

# Streamlit user interface
def main():
    st.title("Speech to Text to PDF Application")
    st.subheader("Record your audio, convert it to text, summarize, and download the summary as a PDF")

    # Input fields
    duration = st.slider("Select recording duration (seconds):", min_value=5, max_value=60, value=30)
    record_button = st.button("Record Audio")

    # Recording the audio
    if record_button:
        audio_data = recording(duration)

        # Saving the recorded audio as a .wav file
        output_audio_file ="temp_audio.wav"
        write(output_audio_file, 44100, audio_data)

        # Convert audio to text
        writtentext = audiototext(output_audio_file)
        summary = clean_text(writtentext)

        st.subheader("Summarized Text:")
        st.write(summary)

        # Generate PDF
        pdf_file = "summary.pdf"
        texttopdf(summary, pdf_file)

        # Provide download link for the generated PDF
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download Summary PDF",
                data=f,
                file_name="summary.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
