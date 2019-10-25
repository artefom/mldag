from sklearn.pipeline import TransformerMixin
import dask.dataframe as dd
import pandas as pd
import os
import logging
from dask.diagnostics import ProgressBar
from pandas.util.testing import assert_frame_equal
import shutil
from .describe import describe, infer_processing
from .preprocessing import preprocess_ds

logger = logging.getLogger(__name__)


class DaskProcessor(TransformerMixin):
    def __init__(self,
                 meta_dir,
                 column_subset=None,
                 partition_col=None):

        assert (partition_col is None or column_subset is None) or \
               (partition_col in column_subset), "Partition column must be in column subset"

        self.column_subset = column_subset
        self.partition_col = partition_col
        self._meta_dir = None

        self._meta_paths_raw = {
            'test_chunk': 'test_chunk.hdf',
            'test_chunk_fitted': 'test_chunk_fitted.hdf',
            'description_raw': 'description_raw.csv',
            'column_meta': 'column_meta.csv',
            'val_meta': 'val_meta.csv',
            'processed': 'processed.parquet',
            'normalized': 'normalized.parquet',
        }
        self._meta_paths = None
        self._meta = None

        self.set_meta_dir(meta_dir)

    def set_meta_dir(self, meta_dir):

        meta_dir = os.path.abspath(meta_dir)
        if self._meta_dir == meta_dir:
            logger.info("Refreshing meta from dir {}".format(meta_dir))
        else:
            logger.info("Setting meta dir {}".format(meta_dir))

        if not os.path.exists(meta_dir):
            os.mkdir(meta_dir)

        # Reset old status
        self._meta_paths = dict()
        self._meta = dict()

        # Set meta paths
        for k, v in self._meta_paths_raw.items():
            self._meta_paths[k] = os.path.join(meta_dir, v)

        # parse meta files from dir
        self.parse_meta_dir()

        self._meta_dir = meta_dir

    def get_meta_dir(self):
        return self._meta_dir

    def refresh_meta(self):
        self.set_meta_dir(self._meta_dir)

    def parse_meta_dir(self):
        for k, v in self._meta_paths.items():
            try:
                self._meta[k] = self._read_meta_file(v)
            except (FileNotFoundError, OSError):
                self._meta[k] = None

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)

    def fit(self, X: dd.DataFrame,
            y=None,
            process_func=None,
            cat_features=None):
        ds = self._get_dataframe(X)
        if y is not None:
            raise NotImplementedError()
        if process_func is None:
            raise ValueError("Please, specify process func")

        # Check if processing function changed
        processing_changed = False
        try:
            self._test_processing_func(process_func)
            logger.info("Processing seems the same")
        except ValueError:
            processing_changed = True
            logger.info("Processing changed!")

        intermediate_file = self._meta_paths['processed']
        if processing_changed:
            self._fix_processing_func(process_func)
            self._test_processing_func(process_func)

            # Create intermediate full processed dataset
            logger.info("Applying processing func")
            with ProgressBar():
                processed_ds = ds.map_partitions(process_func)

            if os.path.exists(intermediate_file):
                shutil.rmtree(intermediate_file)
            processed_ds.to_parquet(intermediate_file)
            self.refresh_meta()

        processed_ds = self.get_processed_ds()

        val_meta, column_meta = self.get_description()
        if processing_changed and (val_meta is None or column_meta is None):
            self._create_description(processed_ds, cat_features=cat_features)

        val_meta, column_meta = self.get_description()

        if processing_changed or self.get_normalized_ds() is None:
            preprocess_ds(
                ds=processed_ds,
                val_meta=val_meta,
                column_meta=column_meta,
                out_fname=self._meta_paths['normalized'])
            self.refresh_meta()

        return self.get_normalized_ds()

    def transform(self, X, y=None):
        if y is not None:
            raise NotImplementedError()
        pass

    @staticmethod
    def _read_meta_file(file: str):
        ext = os.path.splitext(file)[1].lower()
        if ext == '.csv':
            return pd.read_csv(file)
        if ext == '.hdf':
            rv = pd.read_hdf(file, 'df')
            return rv
        if ext == '.parquet':
            rv = pd.read_parquet(file)
            return rv
        raise ValueError(f"Extension {ext} not recognized")

    @staticmethod
    def _read_file(file: str):
        ext = os.path.splitext(file)[1].lower()
        if ext == '.csv':
            return dd.read_csv(file)
        if ext == '.hdf':
            rv = dd.read_hdf(file, 'df')
            return rv
        if ext == '.parquet':
            rv = dd.read_parquet(file)
            return rv
        raise ValueError(f"Extension {ext} not recognized")

    def _get_dataframe(self, X, *args, **kwargs) -> dd.DataFrame:
        logger.info("Reading dataframe")
        if isinstance(X, dd.DataFrame):
            rv = X
        else:
            raise ValueError("X must be dask DataFrame")

        # Ensure proper partition column
        with ProgressBar():
            if self.partition_col is not None and rv.index.name != self.partition_col:
                rv = rv.reset_index().set_index(self.partition_col)

        # If column subset is defined, apply it
        if self.column_subset is not None:
            # Make sure we don't include partition column since it's already in the index
            col_subset_no_partition = [i for i in self.column_subset if i != self.partition_col]
            rv = rv[col_subset_no_partition]

        logger.info("Finished")
        return rv

    def get_processed_ds(self):
        path = self._meta_paths['processed']
        if path is None:
            return None
        try:
            return dd.read_parquet(path)
        except OSError:
            return None

    def get_normalized_ds(self):
        path = self._meta_paths['normalized']
        if path is None:
            return None
        try:
            return dd.read_parquet(path)
        except OSError:
            return None

    def create_test_chunk(self, X):
        ds = self._get_dataframe(X)
        logger.info("Calculating dataset length")
        with ProgressBar():
            part_len = ds.get_partition(0).shape[0].compute()
        logger.info("Creating test chunk")
        with ProgressBar():
            test_chunk = ds.get_partition(0).sample(frac=100 / part_len,
                                                    random_state=42).compute()
        fname = self._meta_paths['test_chunk']
        test_chunk.to_hdf(fname, 'df')
        self.refresh_meta()

    def get_test_chunk(self):
        return self._meta['test_chunk']

    def get_description_raw(self):
        return self._meta['description_raw']

    def get_description(self):
        return self._meta['val_meta'].set_index('column'), self._meta['column_meta'].set_index('column')

    def _create_description(self, ds, cat_features=None):
        description_raw = describe(ds)
        description_raw.to_csv(self._meta_paths['description_raw'])
        val_meta, column_meta = infer_processing(ds, ds_descr=description_raw, cat_features=cat_features)
        column_meta.to_csv(self._meta_paths['column_meta'])
        val_meta.to_csv(self._meta_paths['val_meta'])
        self.refresh_meta()

    def _fix_processing_func(self, func):
        logger.info("Fixing processing func")
        in_ds = self.get_test_chunk()
        out_ds = func(in_ds)
        fname = self._meta_paths['test_chunk_fitted']
        out_ds.to_hdf(fname, 'df')
        self.refresh_meta()

    def _test_processing_func(self, func):
        if self._meta['test_chunk_fitted'] is None:
            raise ValueError("Should first call create_test_chunk")
        in_ds = self.get_test_chunk()
        got = func(in_ds)
        expected = self._meta['test_chunk_fitted']
        assert_frame_equal(got, expected)
