My thesis developes a PDF chatbot, which extract information from PDF files that uploaded by users, uses it as data and then answer questions about these PDF files. 

The code is structured as follows:
- app.py:
  + Contains utility functions for PDF text extraction, chunking, and vector store construction, conversational retrieval chain building.
  + Contains functions for handling user request  and display chatbot responses
- htmlTemplates.py: contains HTML templates for displaying chat history and avatar.
- requirements.txt: contains all basic required libraries.
  
How to use:
- First of all, download the llama 2 7B GGUF model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf and place it in the models folder.
- Run the command pip install `pip install -r requirements.txt` to install dependencies
- Run the command  `streamlit run app.py` to run the application
- Upload PDF files, click on Process button, wait until the system processes these files successfully and then ask questions about these files.
