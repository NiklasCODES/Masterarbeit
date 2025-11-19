
import pandas as pd

class DataFrameProcessor:
    def __init__(self, df: pd.DataFrame, exclude_columns_from_filter, date_column, target, label, graph, withGraph=True):
        '''
        Args:
            df: the dataframe to process
            exclude_columns_from_filter: a list of columns to exclude from filtering, by default, all columns
                are used for filtering
            date_column: the name of the date column for time series data
            target: the name of the target column to aggregate
            label: A label for the data, this is for differentiating the actual data with prediction graphs in the legend
            graph: The column name that differentiates the rows for the graphs.
        '''
        self.target = target
        self.date_column = date_column
        self.label = label
        self.graph = graph
        self.withGraph = withGraph
        self.df = df.copy()
        self.df.reset_index(inplace=True)
        filter_columns = list(df.columns.values)
        self.filter_columns = [c for c in filter_columns if c not in exclude_columns_from_filter]
        self.df[date_column] = pd.to_datetime(self.df[date_column].astype(str), format='%Y%m%d')
        if self.withGraph:
            self.df[self.graph] = label
        self.min_date = self.df[date_column].min()
        self.max_date = self.df[date_column].max()

    def reset_df(self, df):
        '''
        Resetting the dataframe. This is good after aggregating and filtering, if the UI is updated with filters.
        Args:
            df: the dataframe
        '''
        self.df = df.copy()
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column].astype(str), format='%Y%m%d')
        if self.withGraph:
            self.df[self.graph] = self.label


    def insert_empty_row_at_pos(self, pos):
        # Insert the empty row
        empty_row = {col: None for col in self.df.columns}
        self.df = pd.concat([
            self.df.iloc[:pos],
            pd.DataFrame([empty_row]),
            self.df.iloc[pos:]
        ], ignore_index=True)
        self.df.reset_index(drop=True, inplace=True)

    def change_target(self, target):
        '''
        Changing the target column, that is used for predictions.
        Args:
            target: the name of the target column.
        '''
        self.target = target

    def filter_by_specified_columns(self, filters, inplace=False):
        '''
        Filter by specified columns.
        Args:
            filters: a dict with column names as keys and values to filter.
                Example: "Niederlassung": "NL Dortmund"
        '''
        mask = pd.Series(True, index=self.df.index)
        for col, val in filters.items():
            if val not in [None, "", []]:
                if isinstance(val, list):
                    mask &= self.df[col].isin(val)
                else:
                    mask &= self.df[col] == val

        filtered_df = self.df[mask]

        if inplace:
            self.df = filtered_df
        else:
            return filtered_df

    def aggregate(self, inplace=True):
        '''
        Aggregates based on dates.
        '''
        data = (
            self.df
            .groupby([self.date_column], as_index=False)[self.target]
            .sum()
        )
        data[self.graph] = self.target

        if inplace:
            self.df = data.copy()
        else:
            return data

    def aggregate_by_timespan(self, start_date, end_date, inplace=False):
        '''
        Sums together the results, based on the current timespan, for better
        visibility of the data.
        Args:
            start_date: start date
            end_date: end date
        '''
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        days = (end - start).days
    
        if days > 365:
            data = self.df.resample("M", on=self.date_column)[self.target].sum().reset_index()
        elif days > 90:
            data = self.df.resample("W", on=self.date_column)[self.target].sum().reset_index()
        else:
            data = self.df.resample("D", on=self.date_column)[self.target].sum().reset_index()

        data[self.graph] = self.target
        if inplace:
            self.df = data
        else:
            return data

    def add_prediction(self, label, last_n_rows, predict, inplace=False):
        '''
        As typical with time series data, n "lags" (last rows) are used for prediction.
        The prediction is added to the dataframe with a new "graph type" that is set.
        This is convenient for displaying the dataframe in a line chart, where different
        graphs can be displayed with a "color" column.
        Args:
            label: the name of the prediction
            last_n_rows: the last n rows, that should be used for the prediction,
                the prediction itself will then return a vector with n entries
            predict: a function that receives a dataframe and returns a series
        '''
        if not self.withGraph:
            return self.df

        rows = self.df[self.df[self.graph] == self.target].tail(last_n_rows).copy()

        rows[self.target] = predict(rows)
        rows[self.graph] = label 

        data = pd.concat([self.df, rows], ignore_index=True)

        if inplace:
            self.df = data
        else:
            return data

    def filter_by_date(self, start_date, end_date, inplace=False):
        '''
        Args:
        '''
        mask = (
            (self.df[self.date_column] >= start_date)
            & (self.df[self.date_column] <= end_date)
        )

        if inplace:
            self.df = self.df.loc[mask]
        else:
            return self.df.loc[mask]

