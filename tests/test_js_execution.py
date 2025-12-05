
import gradio as gr

# JS to turn the box green
js_code = """
<script>
console.log("🚀 Script Injected via HTML");
document.addEventListener('DOMContentLoaded', () => {
    console.log("⚡ DOMContentLoaded");
    setTimeout(() => {
        const box = document.getElementById('js-status');
        if (box) {
            box.style.backgroundColor = '#00ff00';
            box.innerText = '✅ JS IS RUNNING';
            console.log("✅ Box updated");
        } else {
            console.error("❌ Box not found");
        }
    }, 1000);
});
</script>
"""

def process(audio):
    return audio

with gr.Blocks(title="JS Execution Test") as demo:
    gr.Markdown("## 🧪 JS Execution Test")
    
    # 1. The Test Box
    gr.HTML("""
    <div id="js-status" style="width: 100%; height: 100px; background-color: red; color: white; font-size: 24px; font-weight: bold; display: flex; align-items: center; justify-content: center; border: 4px solid black;">
        ❌ JS NOT RUNNING
    </div>
    """)
    
    # 2. Injection Attempt 1: Direct HTML
    gr.HTML(js_code)
    
    with gr.Row():
        inp = gr.Audio(label="Input")
        out = gr.Audio(label="Output")
    
    inp.change(process, inp, out)

if __name__ == "__main__":
    demo.launch()
