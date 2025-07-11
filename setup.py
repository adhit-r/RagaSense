from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="raga-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning system for detecting Indian classical music ragas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/raga-detector",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    install_requires=[
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "numpy==1.24.3",
        "librosa==0.10.0.post2",
        "scikit-learn==1.3.0",
        "tensorflow==2.13.0",
        "python-dotenv==1.0.0",
        "Werkzeug==2.3.7",
        "soundfile==0.12.1",
        "pydub==0.25.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
