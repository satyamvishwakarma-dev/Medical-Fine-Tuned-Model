import gradio as gr


with gr.Blocks() as app:
    gr.Markdown("# Medical Fine-Tuned Model")

    input_box = gr.Textbox(label="Input Text")
    output_box = gr.Textbox(label="Output Text")

    input_box.change(fn=text, inputs=input_box, outputs=output_box)

app.launch()