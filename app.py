import streamlit as st
import tempfile
from main import build_vectorstore, answer_question

st.write("APP IS RUNNING") 

st.title('RAG Based Document Analysis tool')

st.markdown(
    "<p style='font-size: 30px;'>This app allows you to upload your documents and chat with your PDF.</p>",
    unsafe_allow_html=True
)


# CSS to hide the user avatar — top level, runs once, applies to whole page
st.markdown("""
<style>
[data-testid="stChatMessageAvatarUser"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.info("Note: Running on free-tier CPU, so document processing takes ~1 minute. Thanks for your patience!")
uploaded_file = st.file_uploader('Upload a PDF file', type="pdf")

if uploaded_file is not None:

    if st.session_state.get("processed_file") != uploaded_file.name:
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
      with st.spinner("Processing document..."):
        build_vectorstore(tmp_path) 
      
      st.session_state.processed_file = uploaded_file.name
      st.success("Document ready!")


    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 3. redraw the conversation so far
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question")

    if query:

        st.session_state.messages.append({"role": "user" , "content":query})
        with st.chat_message("user"):
           st.write(query)
                
        response = answer_question(query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
          st.write(response)


    