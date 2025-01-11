import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="Career Development Portal",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    ]
if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False

# API endpoint
API_BASE_URL = "http://localhost:8000"  # Adjust according to your FastAPI server


def login_user(email, password):
    try:
        response = requests.post(f"{API_BASE_URL}/login", params={"email": email, "password": password})
        if response.status_code == 200:
            st.session_state.user_email = email
            return True
        return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False


def register_user(email, password, full_name):
    try:
        response = requests.post(
            f"{API_BASE_URL}/register",
            json={"email": email, "password": password, "full_name": full_name}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error during registration: {str(e)}")
        return False


def upload_resume(file, career_path):
    try:
        files = {"file": ("resume.pdf", file, "application/pdf")}

        # Send email and career_goal as query parameters
        params = {
            "email": st.session_state.user_email,
            "career_goal": career_path
        }

        # Make the POST request with query parameters
        response = requests.post(
            f"{API_BASE_URL}/upload-resume",
            params=params,  # Use params for query parameters
            files=files
        )
        st.session_state.resume_analyzed = True
        return response.json()
    except Exception as e:
        st.error(f"Error uploading resume: {str(e)}")
        return None


def get_chat_response():
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            params={"chat_history": json.dumps(st.session_state.chat_messages)}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error getting chat response: {str(e)}")
        return None


def chat_interface(summary):
    st.subheader("Career Assistant Chatbot")

    if len(st.session_state.chat_messages)==1:
        st.session_state.chat_messages.append({"role": "user", "content": summary})

        st.session_state.chat_messages.append({"role": "assistant", "content": ""})

    # Display chat messages (skip the system message)
    for message in st.session_state.chat_messages[1:]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your career development..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)


        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response()
                if response:
                    st.write(response)
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})

                    if len(st.session_state.chat_messages) > 5:
                        st.session_state.chat_messages.pop(2)
                else:
                    st.error("Failed to get response from assistant")


def main():
    st.title("Career Development Portal 🚀")

    # Sidebar for login/register
    with st.sidebar:
        st.header("Account Management")
        if not st.session_state.user_email:
            tab1, tab2 = st.tabs(["Login", "Register"])

            with tab1:
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login"):
                    if login_user(email, password):
                        st.success("Logged in successfully!")
                        st.rerun()

            with tab2:
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                full_name = st.text_input("Full Name")
                if st.button("Register"):
                    if register_user(reg_email, reg_password, full_name):
                        st.success("Registration successful! Please login.")
        else:
            st.write(f"Logged in as: {st.session_state.user_email}")
            if st.button("Logout"):
                st.session_state.user_email = None
                st.session_state.chat_messages = [
                    {"role": "system",
                     "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
                ]
                st.session_state.resume_analyzed = False
                st.rerun()

    # Main content
    if st.session_state.user_email:
        st.header("Resume Analysis & Career Assistant")

        # Resume upload section
        if not st.session_state.resume_analyzed:
            career_goal = st.text_input("Enter your desired career goal", key="career_goal")
            uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
            if career_goal and uploaded_file and st.button("Analyze Resume"):
                with st.spinner("Analyzing resume..."):
                    result = upload_resume(uploaded_file, career_goal)
                    if result:
                        st.success("Resume analyzed successfully!")
                        st.session_state.score = result.get('resume_score', '')

                        st.session_state.resume_analyzed = True
                        st.session_state.summary = result.get("summary","")
                        st.rerun()

        # Show analysis results and chat interface after resume is analyzed
        if st.session_state.resume_analyzed:
            # You can add a section here to display the resume analysis results
            st.subheader("Resume Analysis Results")
            st.write(
                f"<h2 style='color:green;'>Career Goal Progress: {st.session_state.score}</h2>",
                unsafe_allow_html=True
            )
            st.info("Your resume has been analyzed. You can now chat with the assistant about your career development.")

            # Divider between analysis and chat
            st.markdown("---")

            # Chat interface
            chat_interface(st.session_state.summary)


if __name__ == "__main__":
    main()