import torch
import csv
import pandas as pd
import numpy as np

from collections import deque


# hardcoding maximum entries in the time series observation to 20
max_entries = 10

def get_time(time_string):

    if time_string == "MORNING":
        return 0
    elif time_string == "EVENING":
        return 0.5
    else:
        print("Invalid Time string")
        return 0

class TimeSeriesData:

    def __init__(self, name, intial_pain_scale, initial_mood_scale, 
                 recorded_time, time_string) -> None:
        
        self.pain_scale_entries = deque(np.zeros(max_entries))
        self.mood_scale_entries = deque(np.zeros(max_entries))

        self.time_diff_entries  = deque(np.zeros(max_entries))
        self.input_flags        = deque(np.zeros(max_entries))
        
        self.pain_scale_entries.append(float(intial_pain_scale/10))
        self.pain_scale_entries.popleft()

        self.mood_scale_entries.append(float(initial_mood_scale/10))
        self.mood_scale_entries.popleft()

        self.input_flags.append(float(1))
        self.input_flags.popleft()

        self.initial_day = pd.to_datetime('1900-01-01 00:00:00')

        self.previous_record_time = float((recorded_time - self.initial_day).days + get_time(time_string))

    def update(self, pain_scale, mood_scale, recorded_time, time_string, progression, timing):

        self.pain_scale_entries.append(float(pain_scale/10))
        self.pain_scale_entries.popleft()

        self.mood_scale_entries.append(float(mood_scale/10))
        self.mood_scale_entries.popleft()

        self.current_record_time = float((recorded_time - self.initial_day).days + get_time(time_string))

        time_diff = self.current_record_time - self.previous_record_time
        
        self.time_diff_entries.append(float(time_diff)/14)
        self.mood_scale_entries.popleft()

        self.input_flags.append(float(1))
        self.input_flags.popleft()

        self.previous_record_time =  self.current_record_time
        self.progression =  float(progression)
        self.timing      =  float(timing)


    def get_features(self):
        
        pain_entries = list(self.pain_scale_entries)
        mood_entries = list(self.mood_scale_entries)
        time_diffs   = list(self.time_diff_entries)
        input_flags  = list(self.input_flags)

        return pain_entries + mood_entries + time_diffs + input_flags
        

    def get_target(self):
        return self.progression, self.timing
    


if __name__ == "__main__":

    excel_data = pd.read_excel('data\Data for model.xlsx', index_col=0)  
    
    # group data based on user ids
    gk = excel_data.groupby('user_id')
    groups = dict(list(gk))

    time_series_data = r'data\time_series_data.csv'

    pain_entries_header = [f"pain_scale_{i}" for i in range(max_entries)]
    mood_entries_header = [f"mood_scale_{i}" for i in range(max_entries)]
    time_diff_header    = [f"time_diff_{i}" for i in range(max_entries)]
    input_flags_header  = [f"input_flag_{i}" for i in range(max_entries)]

    feature_header      = pain_entries_header + mood_entries_header + time_diff_header + input_flags_header
    target_header       = ['progression', 'timing']


    with open(time_series_data, 'w',  newline='') as csv_file:
        
        writer = csv.writer(csv_file)
        header = ['Patient ID'] + feature_header + target_header
        writer.writerow(header)
        
        for name, df in groups.items():
            if len(df) >= 2:
                time_series_data = TimeSeriesData(name, 
                                                df["pain_scale"].iloc[0], 
                                                df["mood_scale"].iloc[0], 
                                                df['date'].iloc[0],  
                                                df['time_of_day'].iloc[0])
                    

                for i in range(1, df.shape[0]):
                    time_series_data.update(df["pain_scale"].iloc[i], 
                                            df["mood_scale"].iloc[i], 
                                            df['date'].iloc[i],  
                                            df['time_of_day'].iloc[i],
                                            df['Progression'].iloc[i],
                                            df['Timing'].iloc[i])

                    xi_features               = time_series_data.get_features()
                    yi_progression, yi_timing = time_series_data.get_target()
                    
                    data_row = [name] + xi_features + [yi_progression, yi_timing]
                    writer.writerow(data_row)

    print('Time series data generation finished!')
