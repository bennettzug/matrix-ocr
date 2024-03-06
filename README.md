# matrix-ocr


https://github.com/bennettzug/matrix-ocr/assets/42450925/27b9416f-d789-4274-83e0-e3fc10bd87b9

Simple script to row reduce matrices from OCR of screenshots in the clipboard. The actual row reduction is built with sympy, a Python symbolic mathematics library, and the non-OCR code should be pretty extensible.

### How to install
this script *currently* requires tesseract, [brew](https://brew.sh/) install tesseract then set up the venv with requirements.txt follwoing [these steps](https://stackoverflow.com/questions/41427500/creating-a-virtualenv-with-preinstalled-packages-as-in-requirements-txt) then you should be able to call test.py.

If you're on Windows or Linux (or if you installed tesseract a different way), the code will have issues finding tesseract, but it should be pretty obvious what to change (point the tesseract_cmd line to your installation)  
