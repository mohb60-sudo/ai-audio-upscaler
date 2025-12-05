
import gradio as gr

def process(audio):
    return audio

with gr.Blocks(title="JS Execution Test V3") as demo:
    gr.Markdown("## 🧪 JS Execution Test V3 (The Nuclear Option)")
    gr.Markdown("This test uses an invalid image to trigger an error handler, which executes JavaScript. This bypasses most script filters.")
    
    # Image with onerror - if this doesn't alert, JS is 100% blocked
    gr.HTML("""
    <div style="border: 2px solid blue; padding: 20px; text-align: center;">
        <h3>Image OnError Test</h3>
        <!-- This image is invalid, so it triggers onerror immediately -->
        <img src="invalid_image_to_trigger_js.jpg" 
             style="display: none;"
             onerror="
                console.log('🚀 OnError Triggered');
                document.getElementById('status').innerText = '✅ JS RUNNING (via onerror)';
                document.getElementById('status').style.backgroundColor = '#00ff00';
             " 
        />
        <div id="status" style="background-color: red; color: white; padding: 20px; font-size: 24px; font-weight: bold;">
            ❌ JS NOT RUNNING
        </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
