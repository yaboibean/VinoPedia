import streamlit as st
import streamlit.components.v1 as components

# Read the HTML file
with open("index.html", "r") as f:
    html_code = f.read()

# Render the HTML/JS/CSS exactly as in index.html
components.html(html_code, height=900, scrolling=True)

