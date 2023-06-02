import requests
from PIL import Image
from io import BytesIO
from fake_useragent import UserAgent as ua
import json
import modules.scripts as scripts
import gradio as gr
import os
from modules import script_callbacks
import time
import threading
import urllib.request
import urllib.error
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError
import urllib.request

MODELFOLDER = "extensions\sd-webui-controlnet\models"
DOWNLOAD_STATUS = "Please wait..."

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            model_name = gr.Radio(
                label='Select model:',
                choices=["Canny","Depth","Normal","OpenPose", "MLSD", "Lineart", "SoftEdge", "Scribble", "Seg", "Shuffle", "Tile", "Inpaint", "IP2P"],
                value="none",
                type="value")
        with gr.Row():
            with gr.Column(scale=1):
                download_model = gr.Button(
                    "Download Model"
                ).style(
                    full_width=True
                )
            with gr.Column(scale=6):
                preview_html = gr.TextArea(interactive=False, lines=1, show_label=False, visible=False)
        with gr.Row():
            pr = gr.Progress(50)

        download_model.click(
            fn=get_model_url,
            inputs=[model_name],
            outputs=[preview_html]
        )

        return [(ui_component, "ControlNet Models", "extension_cn_models")]

def get_model_url(model_name):
    if model_name == "none":
        DOWNLOAD_STATUS = "Please select a controlnet model"
        return gr.TextArea.update(value="Please select a controlnet model", visible=True)
    else:
        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/"
        if model_name == "Canny":
            file_name = "control_v11p_sd15_canny.pth"
        elif model_name == "Depth":
            file_name = "control_v11f1p_sd15_depth.pth"
        elif model_name == "Normal":
            file_name = "control_v11p_sd15_normalbae.pth"
        elif model_name == "OpenPose":
            file_name = "control_v11p_sd15_openpose.pth"
        elif model_name == "MLSD":
            file_name = "control_v11p_sd15_mlsd.pth"
        elif model_name == "Lineart":
            file_name = "control_v11p_sd15_lineart.pth"
        elif model_name == "SoftEdge":
            file_name = "control_v11p_sd15_softedge.pth"
        elif model_name == "Scribble":
            file_name = "control_v11p_sd15_scribble.pth"
        elif model_name == "Seg":
            file_name = "control_v11p_sd15_seg.pth"
        elif model_name == "Shuffle":
            file_name = "control_v11e_sd15_shuffle.pth"
        elif model_name == "Tile":
            file_name = "control_v11f1e_sd15_tile.pth"
        elif model_name == "Inpaint":
            file_name = "control_v11p_sd15_inpaint.pth"
        elif model_name == "IP2P":
            file_name = "control_v11e_sd15_ip2p.pth"
        else:
            file_name = "control_v11p_sd15_canny.pth"
        
        url += file_name
        msg = "Downloading "+file_name+"..."
        DOWNLOAD_STATUS = msg
        
        download_file_thread(url, file_name)

        return gr.TextArea.update(value=msg, visible=True)

def download_file(url, file_name, pr=gr.Progress(track_tqdm=True)):
    if not os.path.exists(MODELFOLDER):
        os.makedirs(MODELFOLDER)

    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10

    while True:
        # Check if the file has already been partially downloaded
        if os.path.exists(file_name):
            # Get the size of the downloaded file
            downloaded_size = os.path.getsize(file_name)

            # Set the range of the request to start from the current
            # size of the downloaded file
            headers = {"Range": f"bytes={downloaded_size}-"}
        else:
            downloaded_size = 0
            headers = {}

        # Split filename from included path
        tokens = re.split(re.escape('\\'), file_name)
        file_name_display = tokens[-1]

        # Initialize the progress bar
        progress = tqdm(total=1000000000, unit="B", unit_scale=True,
                        desc=f"Downloading {file_name_display}",
                        initial=downloaded_size, leave=False)

        # Open a local file to save the download
        with open(file_name, "ab") as f:
            while True:
                try:
                    # Send a GET request to the URL and save the response to the local file
                    response = requests.get(url, headers=headers, stream=True)

                    # Get the total size of the file
                    total_size = int(response.headers.get("Content-Length", 0))

                    # Update the total size of the progress bar if the `Content-Length` header is present
                    if total_size == 0:
                        total_size = downloaded_size
                    progress.total = total_size

                    # Write the response to the local file and update the progress bar
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            progress.update(len(chunk))

                    downloaded_size = os.path.getsize(file_name)
                    # Break out of the loop if the download is successful
                    break
                except ConnectionError as e:
                    # Decrement the number of retries
                    max_retries -= 1

                    # If there are no more retries, raise the exception
                    if max_retries == 0:
                        raise e

                    # Wait for the specified delay before retrying
                    time.sleep(retry_delay)

        # Close the progress bar
        progress.close()
        downloaded_size = os.path.getsize(file_name)
        # Check if the download was successful
        if downloaded_size >= total_size:
            DOWNLOAD_STATUS = "{file_name_display} successfully downloaded."
            print(f"{file_name_display} successfully downloaded.")
            gr.TextArea(value="{file_name_display} successfully downloaded.", visible=True)
            break
        else:
            DOWNLOAD_STATUS = "Error: File download failed. Retrying... {file_name_display}"
            print(f"Error: File download failed. Retrying... {file_name_display}")
            gr.TextArea(value="Error: File download failed. Retrying... {file_name_display}", visible=True)


def download_file_thread(url, file_name):
    path_to_new_file = os.path.join(MODELFOLDER, file_name)
    
    print(path_to_new_file)

    thread = threading.Thread(target=download_file, args=(url, path_to_new_file))

    # Start the thread
    thread.start()

script_callbacks.on_ui_tabs(on_ui_tabs)
