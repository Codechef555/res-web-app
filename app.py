import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF resume
# Remove unwanted characters and extra white spaces
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    text = " ".join(text.split())
    return text

# Load the spacy model 
nlp = spacy.load("en_core_web_sm")

# Create a TF-IDF vectorizer with cosine similarity
vectorizer = TfidfVectorizer(stop_words="english")

# Create the web interface using Streamlit
st.title("Resume shortlisting")

# Input fields for name, email, and PDF file upload
name = st.text_input("Enter your name:")
email = st.text_input("Enter your email:")
pdf_file = st.file_uploader("Upload a PDF file")

# Dropdown menu for qualification
st.subheader("Qualification")
qualification = st.selectbox("Select your qualification:", ["B.Tech", "BCA", "M.Tech", "None"])

# Input field for JD
st.subheader("Job Description")
job_description = st.text_area("Enter the job description for a software engineer:")

# If a PDF file and job description are uploaded, match them and display the result or display unqualified
if pdf_file is not None and job_description != "" and qualification != "None":
    # Convert PDF to clean text
    clean_text = extract_text_from_pdf(pdf_file)
    # Create a spaCy doc object for the job description
    job_description_doc = nlp(job_description)
    # Create a spaCy doc object for the clean text
    clean_text_doc = nlp(clean_text)
    # Get the TF-IDF vectors for the job description and the clean text
    job_description_vector = vectorizer.fit_transform([job_description])
    clean_text_vector = vectorizer.transform([clean_text])
    # Calculate the cosine similarity between the job description and the clean text vectors
    similarity = cosine_similarity(job_description_vector, clean_text_vector)
    # Display the similarity score
    # Determine if the resume is qualified or not based on the similarity score
    st.subheader("Match Score:")
    st.write(similarity[0][0])
    if similarity[0][0] >= 0.7:
        st.subheader("Qualified")
        st.write(f"{name}, resume matches the job description!")
    else:
        st.subheader("Not Qualified")
        st.write(f"Sorry {name}, resume does not match the job description.")
elif qualification == "None":
    st.subheader("Not Qualified")
    st.write(f"Sorry {name}, the candidate with username '{email}' does not meet job requirements because qualification is '{qualification}'.")
else:
    st.write("fill the blanks to generate the output")
