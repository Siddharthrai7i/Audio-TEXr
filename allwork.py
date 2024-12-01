import streamlit as st
import speech_recognition as sr
import pyaudio
import wave
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

# Recording using pyaudio
def recording(duration, samplerate=44100, channels=1, output_file="temp_audio.wav"):
    st.write(f"Recording for {duration} seconds...")
    
    p = pyaudio.PyAudio()
    
    # Open a stream to record audio
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=1024)
    
    frames = []
    for _ in range(0, int(samplerate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save audio data to a .wav file
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(samplerate)
        wf.writeframes(b''.join(frames))
    
    st.write("Recording completed!")
    return output_file

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
        audio_file = recording(duration)

        # Convert audio to text
        writtentext = audiototext(audio_file)
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
