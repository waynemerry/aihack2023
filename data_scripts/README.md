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

We are using private private VAFA sourced injury data licensed to the charity communities and made available by them to the Hackathon team. Contact 
waynemerry@greatcommunities.com.au for accessing the data.  The data available in Postgres databases which can be accessed through a docker container. We filtered and reformatted into an an excel file. 

Save the data under folder named data in repository.

```bash
$ cd aihack2023
$ mkdir data
```

We decided to model **Progression**and **Timing** information from the data. **Progression** is a Boolean value represents  patient's current wellbeing with respect to historical records. **Timing** is a Boolean value represents whether the patient's recording intervals are enough based on patient's current wellbeing. We are using historical pain, mood and record interval time of the patient to predict Progression and Timing. The reformatted excel data contains all patients records which need to be converted to time series data as Progression and Timing are depends on historical records of the patient. 

Run following script for time series data generation. 

```bash
$ python data_scripts\timeseries_data_generation.py
```

This creates a csv file with time series data of pain, mood and records interval with progression and timing labels in data folder.  We trained binary classification models for Progression and Timing separately.  The models are neural networks with linear, relu and sigmoid activation functions (defined in file data_scripts\models.py ). 

Run following script for training progression model.

```bash
$ python data_scripts\train_progression.py
```

Run following script for training timing model.

```bash
$ python data_scripts\train_timing.py
```

We tested different model architectures by changing  depth, width and keeping dropout layers but not observed any change in accuracy. We observed that model producing grater than 90% accuracy in in our validation data which is proving the model is good enough for predicting Progression and Timing with historical data. 