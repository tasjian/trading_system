
## NOTE: THIS SERVER IS RUNNING PERPETUALLY FOR THIS COURSE.
## DO NOT CHANGE CODE HERE; INSTEAD, INTERFACE WITH IT VIA USER INTERFACE
## AND BY DEPLOYING ON PORT :9012

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#####################################################################
## Final App Deployment

FRAMEWORK = str(os.environ.get("FRONTEND") or "gradio").lower().strip()
PORT = str(os.environ.get("PORT") or "8999").lower().strip()
ROOT_PATH = os.environ.get("ROOT_PATH") or None

print(f"- {FRAMEWORK = }\n- {PORT = }\n- {ROOT_PATH = }")

assert FRAMEWORK in ("gradio", "chainlit"), f"Unknown framework {FRAMEWORK}"


if FRAMEWORK == "chainlit":
    pass

elif FRAMEWORK == "gradio": 
    import gradio as gr
    from frontend_block import get_demo

    demo = get_demo()
    demo.queue()

    logger.warning("Starting FastAPI app")
    app = FastAPI()

    gr.set_static_paths(paths=["imgs", "slides"])
    app = gr.mount_gradio_app(app, demo, '/', root_path=ROOT_PATH)


@app.route("/health")
async def health():
    return {"success": True}, 200
