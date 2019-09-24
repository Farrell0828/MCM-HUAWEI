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

                unique_clutter_index = [2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                user_cols = ['X', 'Y', 'Altitude', 'Building Height', 'Clutter Index']

                # add_norm
                df['hb'] = df['Height'] + df['Cell Altitude'] - df['Altitude']
                df['d'] = ((df['Cell X'] - df['X'])**2 + (df['Cell Y'] - df['Y'])**2)**(1/2) * 0.001
                df['lgd'] = np.log10(df['d'] + 1)
                df['hv'] = df['hb'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
                df['len'] = df['d'] / np.cos(df['Electrical Downtilt'] + df['Mechanical Downtilt'])
                df['lghb'] = np.log10(df['hb'] + 1)
                
                # add_count
                for col in user_cols:
                    df[col + '_count'] = df[col].map(df[col].value_counts())

                # add_density
                df['n'] = len(df)
                df['area'] = ((df['X'].quantile(0.97) - df['X'].quantile(0.03))
                            * (df['Y'].quantile(0.97) - df['Y'].quantile(0.03)))
                df['density'] = df['n'] / df['area']
                
                # add_index
                cell_clutter_dummy = pd.get_dummies(pd.Categorical(df['Cell Clutter Index'], categories=unique_clutter_index), prefix='CellClutterIndex')
                clutter_dummy = pd.get_dummies(pd.Categorical(df['Clutter Index'], categories=unique_clutter_index), prefix='ClutterIndex')
                df = (df.merge(cell_clutter_dummy, left_index=True, right_index=True)
                        .merge(clutter_dummy, left_index=True, right_index=True))

                x_cols = [col for col in df.columns if col not in ['Cell Index', 'Cell Clutter Index', 'Clutter Index', 'RSRP']]
                df = df.fillna(df.mean())
                input_data = df[x_cols].values
                filesDatas.append(input_data)
        filesDatas = np.concatenate(filesDatas, axis=0)
        
        preprocessed_data['myInput'] = filesDatas.astype(np.float32) 
        print('myInput.shape', filesDatas.shape)
        return preprocessed_data


    def _postprocess(self, data):        
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            results_np = np.array(results)
            results_np[np.isnan(results_np)] = -91.78557
            results = results_np.tolist()
            infer_output["RSRP"] = results
        return infer_output