import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd


class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                df = pd.read_csv(file_content)
                df['hb'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
                df['d'] = ((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)**(1/2) * 0.001
                df['hv'] = df['hb'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
                df['len'] = df['d'] / np.cos(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
                df['lghb'] = np.log10(df['hb'])
                unique_clutter_index = [2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                cell_clutter_dummy = pd.get_dummies(pd.Categorical(df['Cell Clutter Index'], categories=unique_clutter_index), prefix='CellClutterIndex')
                clutter_dummy = pd.get_dummies(pd.Categorical(df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
                df = (df.merge(cell_clutter_dummy, left_index=True, right_index=True)
                        .merge(clutter_dummy, left_index=True, right_index=True))
                x_cols = [col for col in df.columns if col not in ['Cell Index', 'Cell Clutter Index', 'Clutter Index', 'RSRP']]
                input_data = df[x_cols].values
                filesDatas.append(input_data)
        filesDatas = np.concatenate(filesDatas, axis=0)
        print(filesDatas.shape)
        preprocessed_data['input_tensor'] = filesDatas.astype(np.float32)   
        return preprocessed_data


    def _postprocess(self, data):        
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            infer_output["RSRP"] = results
            print(results.shape)
        return infer_output