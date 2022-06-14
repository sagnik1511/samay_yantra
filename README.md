<div align="center">
<h1>SamayYantra</h1>
<h3>Weather Prediction of Beutenberg using time series forcasting with deep learning</h3>
<img width="600px" src="https://socialify.git.ci/sagnik1511/samay_yantra/image?description=1&font=Inter&language=1&name=1&owner=1&pattern=Circuit%20Board&theme=Dark" 
alt="banner">
<h1>Samay Yantra</h1>
<img src="https://forthebadge.com/images/badges/built-with-love.svg">
<img src="https://forthebadge.com/images/badges/made-with-python.svg">
<img src="https://forthebadge.com/images/badges/built-with-science.svg">
<h1>GOALS of the Project</h1>
In ancient days, people used to look at the sky or feel it's humidity and several factors tried to predict the upcoming days which were most of them predicted true to a high accuracy.
Nowadays as we can detect several weather factors, the prediction of weather is becoming too complex to be handled by human brain. Deep learning took up the next step and brought ease to this domain.
Time series forecasting with deep neural networks changes the frontiers. Now we can predict/forecast innumerable future attributes based on historical data. Not only it shared us the predictions, but the data can be taken for more advanced analysis and research.
The secondary goal of the project is to implement cutting edge MLOps to actual real problems.
The ternary goal of the project is to implement this on traditional python OOPS but Jupyter Notebook , so that we can match different real-world ML codebase and learn accordingly.
<h1>Technology</h1>
<h3>1. Pytorch</h3>
<h3>2. Scikit-Learn</h3>
<h3>3. Pandas</h3>
<h3>4. Numpy</h3>
<h3>5. MLFlow</h3>
</div>

# Data Collection Process :

The raw data has been recorded by the Weather station of [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/), Jena, Germany.
Jena Weather dataset is made up of many different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. This dataset covers data from January 1st 2004 to December 31st 2020
The actual data is this a copy which is published for academic purposes as a kaggle dataset,  Link : [kaggle/Weather Station Beutenberg Dataset](https://www.kaggle.com/datasets/mnassrib/jena-weather-dataset)
.
The primary data is stored as a single *.csv* file which is later processed to *processed.csv* file to be taken for training.

### Special Note : 
Data has been stored using **DVC(Data version Control)**, so the repository package can be 
used flexibly without adding the data straight in the repo but fetch from any remote source e.g. **AWS S3**, **GDRIVE**, etc.
For this case, the data has been stored in GDRIVE.


# Directory Structure :

The data follows a strict data science project structure.

    .
    └── root/
        ├──.dvc/
        ├── config/
        ├── mlruns/
        ├── models/
        ├── notebooks/
        ├── results/
        └── src/
            ├── data
            ├── features
            ├── models
            └── visualization
            
# Installation and Usage :
<div align="center"><h1>Installation</h1></div>


1. Create a Virtual Environment : [Tutorial](https://docs.python.org/3/library/venv.html)
2. Clone the repository by running this command.
```shell
git clone https://github.com/sagnik1511/samay_yantra.git
```
3. Open the directory with *cmd*.
4. Copy this command in terminal to install dependencies.
```shell
pip install -r requirements.txt
```
5. Installing the requirements.txt may generate some error due to outdated MS Visual C++ Build. You can fix this problem using [this](https://www.youtube.com/watch?v=rcI1_e38BWs).

# Approach :
1. Go to the root directory using `cd` command.
2. The first step is to download the actual data into the project.Copy and run this command.
```shell
dvc pull
```
3. If you want to run the training process, simply change the configuration in `config/pt_training.yaml` and then run this command . Keep in mind that you have to stay at the root directory.
```shell
python -m src.training.pytorch_trainer
```

4. Further usage will be updated soon...

# Results:
You can visit [reports](https://github.com/sagnik1511/SamayYantra/tree/main/reports) directory where all the runs are stored. Currently, for some privacy issues, the mlflow runs are not shared in here.

<div align="center">
<h1>Thanks for visiting :D</h1>
<h3>Do STAR if you find it useful</h3>
<img src="https://swall.teahub.io/photos/small/185-1857418_the-witcher-21-9-wallpaper-witcher-3.jpg?">
</div>


