
import gradio as gr

# PURE JS Code (No <script> tags)
js_code = """
(function() {
    console.log("🚀 Script Injected via demo.load()");
    
    function updateBox() {
        const box = document.getElementById('js-status');
        if (box) {
            box.style.backgroundColor = '#00ff00';
            box.innerText = '✅ JS IS RUNNING';
            console.log("✅ Box updated");
        } else {
            console.log("⏳ Box not found yet...");
            setTimeout(updateBox, 100);
        }
    }
    
    // Try immediately and on load
    updateBox();
    window.addEventListener('DOMContentLoaded', updateBox);
    setTimeout(updateBox, 1000);
})();
"""

def process(audio):
    return audio

with gr.Blocks(title="JS Execution Test V2") as demo:
    gr.Markdown("## 🧪 JS Execution Test V2")
    
    # 1. The Test Box
    gr.HTML("""
    <div id="js-status" style="width: 100%; height: 100px; background-color: red; color: white; font-size: 24px; font-weight: bold; display: flex; align-items: center; justify-content: center; border: 4px solid black;">
        ❌ JS NOT RUNNING
    </div>
    """)
    
    with gr.Row():
        inp = gr.Audio(label="Input")
        out = gr.Audio(label="Output")
    
    inp.change(process, inp, out)
    
    # Correct Injection: Pure JS string
    demo.load(None, None, None, js=js_code)

if __name__ == "__main__":
    demo.launch()
