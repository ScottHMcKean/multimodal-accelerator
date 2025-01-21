import gradio as gr

def process_text(text):
    return text.upper()

# Create the interface
demo = gr.Interface(
    fn=process_text,          # The function to run
    inputs="text",            # Input type
    outputs="text",           # Output type
    title="Text Uppercaser",  # Interface title
    description="Enter some text and I'll convert it to uppercase!" # Description
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()