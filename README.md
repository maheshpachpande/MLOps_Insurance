# Create conda enviroment
- conda create -p fraud_venv1 python=3.9 -y
- conda activate fraud_venv1/

# Create the GitHub Repository and Integrate with Conda virtual environment
echo "# MLOps_Insurance" >> README.md
- git init
- git add README.md
- git commit -m "first commit"
- git branch -M main
- git remote add origin https://github.com/maheshpachpande/MLOps_Insurance.git
- git push -u origin main

- git remote add origin https://github.com/maheshpachpande/MLOps_Insurance.git
- git branch -M main
- git push -u origin main

# Create the .gitignore Python file and pull into conda environment
- git pull https://github.com/maheshpachpande/MLOps_Insurance.git

# Create template.py for creating a Production-grade folder structure
- echo.>template.py

# Create Setup for project and install requirements
- setup.py
- python setup.py install
- pip install -r requirements.txt

# Logger added


