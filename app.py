from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_plotly_events import plotly_events
from graph import building_graph, Graph, preprocess, visualize_graph
from gencode import write_model_code
import cv2
import numpy as np

st.set_page_config(page_title="Draw a pytorch model", layout="wide")

#variables
GDRAW = 'dot' #use dot or plotly to plot graph
inputs = {} #model options

#state variables
if 'graph' not in st.session_state:
    st.session_state.graph = Graph()

if 'code' not in st.session_state:
    st.session_state.code = ""

if 'notebook' not in st.session_state:
    st.session_state.notebook = ""

if 'img_mask' not in st.session_state:
    st.session_state.img_mask = np.zeros((400, 600), dtype=np.uint8)

#sidebar
with st.sidebar:
    # Specify canvas parameters in application
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.checkbox("Update in realtime", True)
    inputs["model_name"] = st.text_input("Model name: ", 'mymodel')
    inputs["initialization"] = st.selectbox("Select initialization: ", ("xavier", "kaiming", "none"))
    change_form = st.empty()

#main layout
col1, col2 = st.columns(2)
with col1:
    # Create a canvas component
    st.write('Draw your model here:')
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=3,
        stroke_color="#000",
        background_color="#eee",
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    #draw -> shape -> model block
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None: #numpy array (height, width, plane)

        #st.write('Model that you drew:')
        #st.image(canvas_result.image_data, clamp=True)

        img = canvas_result.image_data[:,:,0:3].astype(np.uint8)
        img = preprocess(img) #convert to grayscale

        if np.sum(img == 0) == (img.shape[0] * img.shape[1]):
            st.session_state.graph.reset()
            st.session_state.img_mask = np.zeros((400, 600), dtype=np.uint8)
            st.session_state.code = ""

        img_p = img - st.session_state.img_mask # get new strokes only
        added = building_graph(img_p, st.session_state.graph)
        if added: #new stroke added node or edge, then update graph
            st.session_state.img_mask = img # mask to ignore previous strokes

    if st.session_state.graph.nodes:

        with change_form.container():
            st.write("""## Change a layer""")
            option_change = st.selectbox('Select layer to change', [nd.id for nd in st.session_state.graph.nodes])
            new_layer = st.text_input("Change to layer:", help='e.g.: Conv2d(64, 64, 3)')
            submit_button = st.button(label='go')

        if submit_button:
            #get node change it
            sp = new_layer.find("(")
            ep = new_layer.find(")")
            st.session_state.graph.nodes[option_change].op = new_layer[0:sp]
            st.session_state.graph.nodes[option_change].params = new_layer[sp:ep+1]

        if GDRAW == 'dot':  # draw graph dot viz
            fig = st.session_state.graph.build_dot(sel_nd=option_change)
            st.graphviz_chart(fig)
        else:  # draw graph interactive plotly
            fig = visualize_graph(st.session_state.graph)
            selected_points = plotly_events(fig)
            st.write(selected_points)

        st.session_state.code, st.session_state.notebook = write_model_code(st.session_state.graph, inputs)

with col2:
    if st.session_state.code != "":
        st.write("")  # add vertical space
        st.code(st.session_state.code)
        st.download_button("🐍 Download (.py)", st.session_state.code, "generated-code.py")
        st.download_button("📓 Download (.ipynb)", st.session_state.notebook, "generated-notebook.ipynb")


