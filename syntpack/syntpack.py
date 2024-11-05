import numpy as np
import pandas as pd
import os
from math import floor, ceil

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import ADASYN, SMOTE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.sampling import Condition
from realtabformer import REaLTabFormer

from .utils import cramer_mat

class SynTable:
    def __init__(self, df, target_col, method, target_conditions=None):
        self.df = df
        self.target_col = target_col
        self.method = methods
        self.target_conditions = target_conditions

    def _set_model(self, epochs, batch_size, log_frequency, logging_steps):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.df)

        match self.method:
            case 'smote':
                model = SMOTE()
            case 'adasyn':
                model = ADASYN()
            case 'ctgan':
                model = CTGANSynthesizer(metadata, epochs=epochs, log_frequency=log_frequency, batch_size=batch_size)
            case 'tvae':
                model = TVAESynthesizer(metadata, epochs=epochs, batch_size=batch_size)
            case 'rtf':
                model = REaLTabFormer(model_type='tabular', batch_size=batch_size, logging_steps=logging_steps, epochs=epochs)
            case _:
                print("The method doesn't exist.")

        return model

    def _encode_features(self, cols):
        df_ = self.df.copy()

        label_encoder_dict = {}
        for col in cols:
            label_encoder = LabelEncoder()
            df_[col] = label_encoder.fit_transform(df_[col])
            label_encoder_dict[col] = label_encoder

        return df_, label_encoder_dict

    def synthesize(self, num_samples=None, epochs=300, batch_size=30, log_frequency=False, logging_steps=20):
        if (self.method not in ['smote', 'adasyn']) and (num_samples == None):
            print('CTGAN, TVAE, and REaLTabFormer require num_samples.')
            return None

        if self.method == 'rtf':
            if (batch_size != 8) or (batch_size % 16 != 0):
                batch_size = 32

        model = self._set_model(epochs=epochs, batch_size=batch_size, log_frequency=log_frequency, logging_steps=logging_steps)

        if self.method in ['smote', 'adasyn']:
            cols = list(self.df.select_dtypes(['object']).columns)

            df__, label_encoder_dict = self._encode_features(cols)

            y = df__[self.target_col]
            X = df__.drop(self.target_col, axis=1)

            resampled_X, resampled_y = model.fit_resample(X, y)
            df_ = pd.concat([resampled_X, resampled_y], axis=1)

            del df__

            df_[cols] = df_[cols].astype(np.int64)
            for col in cols:
                encoder = label_encoder_dict[col]
                df_[col] = encoder.inverse_transform(df_[col])

        elif self.method in ['ctgan', 'tvae']:
            if self.target_conditions:
                conditions = []
                filled_rows = 0
                for key in self.target_conditions.keys():
                    if key == list(self.target_conditions.keys())[-1]:
                        condition = Condition(num_rows=(num_samples-filled_rows), column_values={self.y: key})
                    else:
                        num_rows = int(floor(self.target_conditions.get(key) * num_samples))
                        condition = Condition(num_rows=num_rows, column_values={self.y: key})
                        filled_rows += num_rows

                    conditions.append(condition)
                
                model.fit(self.df)
                df_ = model.sample_from_conditions(conditions=conditions)

                del conditions

            else:
                model.fit(self.df)
                df_ = model.sample(num_rows=num_samples)

        else:
            model.fit(self.df, target_col=self.target_col)

            if self.target_conditions:
                df__ = []
                filled_rows = 0
                for key in self.target_conditions.keys():
                    if key == list(self.target_conditions.keys())[-1]:
                        df__.append(model.sample((num_samples-filled_rows), seed_input={self.y: key}))
                    else:
                        num_rows = int(floor(self.target_conditions.get(key) * num_samples))
                        df__.append(model.sample(num_rows, seed_input={self.y: key}))
                        filled_rows += num_rows

                df_ = pd.concat(df__)

                delete df__

        return df_

class SynPlot:
    def __init__(self, df):
        self.df = df

    def _setup_plot(self, num_plots, num_cols, figsize):
        num_main_rows = num_plots // num_cols
        num_add_rows = 1 if num_plots % num_cols > 0 else 0
        num_rows = num_main_rows + num_add_rows

        width, height = figsize

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(width*num_cols, height*num_rows))
        fig.tight_layout()

        if num_plots % num_cols > 0:
            del_ax = num_cols - (num_plots % num_cols)
            for i in range(del_ax):
                axes[-1, -(i + 1).axis('off')]

        return fig, axes, num_rows

    def _save_plot(self, name, format='png'):
        plt.savefig(os.path.join(os.getcwd(), name))

    def _create_plot(self, datatype, cols, num_cols, figsize):
        fig, axes, num_rows = self._setup_plot(len(cols), num_cols, figsize)

        for i, column in enumerate(cols):
            row_index = i // num_cols
            col_index = i % num_cols
            ax = axes[row_index, col_index] if num_rows > 1 else axes[col_index]

            if datatype == 'categorical':
                counts = self.df[column].value_counts()
                counts.plot(kind='barh', color='skyblue', ax=ax)

                for index, value in enumerate(counts):
                    msg = f"{value}, {(value / counts.sum()) * 100:.2f}%"
                    ax.text(value, index, msg, ha='left', va='center', fontsize=10, color='black')

                ax.set_title(f"Horizontal Bar Plot for {column}")
                ax.set_xlabel('Count')
            else:
                values = self.df[column].astype(np.float32)
                ax.boxplot(values, showmeans=True, meanline=True, notch=True)
                ax.set_title(f'Boxplot for {column}')

    def categorical_dist(self, num_cols=2, show=True, save=False, name=None, figsize=(8,6)):
        if name == None:
            name = 'categorical_dist'

        cols = list(self.df.select_dtypes(['object']).columns)
        
        if cols:
            self._create_plot(datatype='categorical', cols=cols, num_cols=num_cols, figsize=figsize)

            if show:
                plt.show()

        if save:
            self._save_plot(name)

    def numerical_dist(self, num_cols=2, show=True, save=False, name=None, figsize=(8,6)):
        if name == None:
            name = 'numerical_dist'

        cols = list(self.df.select_dtypes(include=[np.number]).columns)

        if cols:
            self._create_plot(datatype='numerical', cols=cols, num_cols=num_cols, figsize=figsize)

            if show:
                plt.show()

        if save:
            self._save_plot(name)

    def cramer_corr(self, show=True, title=None, save=False, name=None, figsize=(7,7)):
        if title == None:
            title = "Cramer's Correlation Matrix of Dataframe Input"

        if name == None:
            name = 'cramer_corr'

        cols = list(self.df.select_dtypes(['object']).columns)

        if cols:
            plt.figure(figsize=figsize)
            sns.heatmap(cramer_mat(self.df), annot=True, fmt='.2f')
            plt.title(title)

            if show:
                plt.show()

        if save:
            self._save_plot(name)

    def pearson_corr(self, show=True, title=None, save=False, name=None, figsize=(8,8)):
        if title == None:
            title = "Pearson Correlation Matrix of Dataframe Input"

        if name == None:
            name = 'pearson_corr_corr'

        cols = list(self.df.select_dtypes(include=[np.number]).columns)

        if cols:
            num = self.df[cols].corr(method='pearson')

            plt.figure(figsize=figsize)
            sns.heatmap(num, annot=True, fmt='.2f')
            plt.title(title)

            if show:
                plt.show()

        if save:
            self._save_plot(name)
