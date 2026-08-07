from deep_translator import GoogleTranslator
import gradio as gr

def translate_text(text, target_language):
    try:
        # Translates from auto-detected source to the selected target language
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"

# Get a list of supported languages
languages = list(GoogleTranslator().get_supported_languages())

# Build the User Interface
demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(lines=5, label="Enter Text to Translate"), 
        gr.Dropdown(choices=languages, label="Target Language", value='hindi')
    ],
    outputs=gr.Textbox(lines=5, label="Translated Text"),
    title="Language Translation Tool",
    description="Enter text and select a target language to get the translation."
)

if __name__ == "__main__":
    demo.launch()