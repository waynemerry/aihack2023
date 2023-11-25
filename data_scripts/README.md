### Clone the repository

```bash
$ git clone https://github.com/waynemerry/aihack2023.git
```

### Install dependencies in a virtual environment

Create virtual environment

```
python -m venv aihack  
```

Activate virtual environment

In windows
```
.\aihack\Scripts\Activate.ps1
```
In Linux
```
$ source aihack\Scripts\activate.bash
```

Install packages
```
pip install numpy torch pandas tqdm matplotlib scikit-learn    
```
OR
```
pip install -r data_scripts\requirements.txt
```

### Data

We utilize exclusive injury data sourced from private VAFA, licensed to charitable communities, and provided to the Hackathon team by them. For access to the data, please contact [waynemerry@greatcommunities.com.au](mailto:waynemerry@greatcommunities.com.au). The data is stored in Postgres databases, accessible through a docker container. We have filtered and reformatted it into an Excel file.

Save the data under folder named data in repository.

```bash
$ cd aihack2023
$ mkdir data
```

We have chosen to model Progression and Timing information from the data. Progression is a Boolean value that represents the patient's current well-being in relation to historical records. Timing is a Boolean value that indicates whether the patient's recording intervals are sufficient based on their current well-being. To predict Progression and Timing, we utilize the patient's historical data on pain, mood, and recording interval times.

The reformatted Excel data encompasses all patient records, which must be converted into time series data, as Progression and Timing depend on the patient's historical records.

Run following script for time series data generation. 

```bash
$ python data_scripts\timeseries_data_generation.py
```

This generates a CSV file containing time series data of pain, mood, and record intervals, along with progression and timing labels in the 'data' folder. We trained binary classification models for Progression and Timing separately. The models are neural networks with linear, ReLU, and sigmoid activation functions, as defined in the file 'data_scripts\models.py'. 

### Training

Run following script for training progression model.

```bash
$ python data_scripts\train_progression.py
```

Run following script for training timing model.

```bash
$ python data_scripts\train_timing.py
```

We tested various model architectures by adjusting depth and width while maintaining dropout layers, but did not observe any changes in accuracy. We noticed that the model consistently achieved an accuracy greater than 90% on our validation data, indicating its effectiveness in predicting Progression and Timing using historical data.