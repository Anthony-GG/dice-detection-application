import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("opencv-python")
install("tensorflow")
install("easyOCR")
install("torch")