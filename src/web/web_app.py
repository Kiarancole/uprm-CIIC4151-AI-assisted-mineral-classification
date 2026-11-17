import base64

import streamlit as st
import requests as rq


local_api_url = "https:0.0.0.0:8080" # unsure i gotta check port


def main():
    st.sidebar.markdown("---")
    st.sidebar.header("Pages")

    uploaded_image = st.file_uploader("Upload an image", type="image/*")
    if uploaded_image is not None:
        bytes_data = uploaded_image.getvalue()

        base64_img_string = base64.b64encode(bytes_data)

        st.image(bytes_data)
        st.write(base64_img_string)


if __name__ == "__main__":
    main()