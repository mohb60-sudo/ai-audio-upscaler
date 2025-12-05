from setuptools import setup, find_packages

setup(
    name="ai_audio_upscaler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "soundfile",
        "gradio",
    ],
    entry_points={
        "console_scripts": [
            "ai-upscaler=ai_audio_upscaler.cli:main",
        ],
    },
)
