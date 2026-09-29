import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

categories = learn.dls.vocab
def classify_images(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()

examples = []

intf = gr.Interface(
    fn=classify_images,
    inputs=image,
    outputs=label,
    examples=examples
)

if __name__ == "__main__":
    intf.launch(server_name="0.0.0.0", server_port=7680)