call test_venv/Scripts/activate
python -m build
pip install ./
python test.py