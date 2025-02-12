import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from utils import chat

my_input = st.text_input("Enter Your Question Here")

if my_input:
    answer = chat(question=my_input)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the image in the first column
    with col1:
        st.image(
            "agent_graph.png",
            caption="Agent Graph",
            width=300,
            use_container_width=True,
        )

    # Display the answer in the second column
    with col2:
        st.success(answer)
