import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from ai4bmr_core.log.log import logger

from ..data_models.Dataset import Dataset as DatasetConfig


class BaseDataset(ABC, DatasetConfig):
    def __init__(
            self,
            *,
            base_dir: None | Path | str = None,
            raw_dir: Path = None,
            processed_dir: Path = None,
            force_download: bool = False,
            force_process: bool = False,
            **kwargs,
    ):
        """
        The `BaseDataset` class is an abstract class that provides a template for creating new datasets.
        It consumes the user fields of the DatasetConfig class.

        If `processed_files` is defined on the subclass, caching is enabled and the `processed_dir` is used to store
        the output of the `load` function. If `processed_files` is not defined, caching is disabled.

        If `urls` is defined, downloading is enabled and the `raw_dir` is used to store the downloaded files.

        Args:
            base_dir:
            raw_dir:
            processed_dir:
            force_download:
            force_process:
            **kwargs:
        """

        if (
                base_dir is None
                and raw_dir is not None
                and processed_dir is not None
                and raw_dir.parent != processed_dir.parent
        ):
            logger.warning(
                f"""
                Divergent path configuration. You are saving data in different directories.
                raw_dir: {raw_dir}
                processed_dir: {processed_dir}
                """
            )
        if base_dir is not None and (raw_dir is not None or processed_dir is not None):
            logger.warning(
                f"""
                Paths are over-configured. You are setting `data_dir` and `raw_dir` or `processed_dir` at the same time.
                base_dir (\033[93mignored\033[0m): {base_dir}
                raw_dir (\033[92mused\033[0m): {raw_dir}
                processed_dir (\033[92mused\033[0m): {processed_dir}
                """
            )
        logger.info("Initializing dataset... 🚀")

        # note: we initialize the `BaseDataset` with default values for `raw_dir` and `processed_dir`
        super().__init__(
            # we pass these values directly, because they do not need to be post-processed like the other fields
            force_download=force_download,
            force_process=force_process,
            **kwargs,
        )
        # note: after super(), we can access the initialized values and overwrite them if necessary

        # note: if the user defines a `base_dir` use it. If not, use the default value defined on the subclass.
        #   If the subclass does not define a default value, use the default value defined on the DatasetConfig class.
        self.base_dir = base_dir if base_dir else self._base_dir

        # note: see explanation for `data_dir`
        self.processed_dir = processed_dir if processed_dir else self._processed_dir

        # note: see explanation for `data_dir`
        self.raw_dir = raw_dir if raw_dir else self._raw_dir

        logger.info(
            f"Dataset initialized 🎉 with \n\traw_dir file://{self.raw_dir}\n\tprocessed_dir file://{self.processed_dir}"
        )

    def setup(self):
        logger.info("Setting up dataset... 🫡")
        if self.force_download and self.raw_dir.exists():
            # NOTE: we delete the `raw_dir` if  `force_download` is True to ensure that all files are newly
            #  downloaded and not just some of them in case of an exception occurs during the download.
            #  Furthermore, we check if the folder exists and do not use `is_downloaded` as this is only True if the
            #  previously attempted downloads were successful.
            shutil.rmtree(self.raw_dir)

        if self.force_process and self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)

        if self._urls and (self.force_download or not self.is_downloaded):
            self.download()

        if self.is_cached and not self.force_process:
            logger.info(
                "Cached data found 🥳. Loading data from cache. To re-process pass `force_process=True`."
            )
            self._data = self.load_cache()
        else:
            if self.is_cached:
                logger.info(
                    "Cached data found 🥳 but `force_process=True`. Processing... 🏃🏽"
                )
            else:
                logger.info("No cached data found 😔. Processing... 🏃🏽")
            self._data = self.load()
            self.save_cache(self._data)

        self._is_setup = True
        return self

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    def load_cache(self) -> any:
        # NOTE: handle loading of `processed_files` here
        # files = self.processed_files
        return None

    def save_cache(self, data):
        if self.processed_dir:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
        # add logic to save `processed_files` here

    def download(self):
        # NOTE: this could be a solution to extract the filename from the download URL as fallback solution
        # import requests
        # from urllib.parse import unquote
        #
        # url = "https://zenodo.org/records/4607374/files/SingleCell_and_Metadata.zip?download=1"
        #
        # response = requests.get(url, stream=True)
        # # Ensure the request was successful
        # response.raise_for_status()
        #
        # # Try to extract the filename from the Content-Disposition header
        # content_disposition = response.headers.get('Content-Disposition')
        # if content_disposition:
        #     filename = content_disposition.split('filename=')[-1]
        #     filename = filename.strip('\"\'')  # Remove any enclosing quotes
        #     filename = unquote(filename)  # Decode URL-encoded characters
        # else:
        #     # Fallback to extracting from the URL
        #     filename = url.split('/')[-1].split('?')[0]  # Basic extraction method
        #
        # print(filename)
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            for file_name, url in self._urls.items():
                if (self.raw_dir / file_name).exists():
                    logger.info(
                        f"File {file_name} already exists in {self.raw_dir}. "
                        f"⏭️ Skipping download. 💡For re-download, pass `force_download=True`."
                    )
                    continue
                logger.info(
                    f"💾 Downloading file {file_name} to {self.raw_dir} from {url}"
                )
                self._download_progress(url, self.raw_dir / file_name)
        except Exception as e:
            # if an exception occurs, and the raw_dir is empty, delete it
            # NOTE: there is the case where the raw_dir is not empty, but only some files have been downloaded
            if list(self.raw_dir.iterdir()) == 0:
                shutil.rmtree(self.raw_dir)
            raise e

    @staticmethod
    def _download_progress(url: str, fpath: Path):
        from tqdm import tqdm
        from urllib.request import urlopen, Request

        blocksize = 1024 * 8
        blocknum = 0

        try:
            with urlopen(Request(url, headers={"User-agent": "dataset-user"})) as rsp:
                total = rsp.info().get("content-length", None)
                with tqdm(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        unit_divisor=1024,
                        total=total if total is None else int(total),
                ) as t, fpath.open("wb") as f:
                    block = rsp.read(blocksize)
                    while block:
                        f.write(block)
                        blocknum += 1
                        t.update(len(block))
                        block = rsp.read(blocksize)
        except (KeyboardInterrupt, Exception):
            # Make sure file doesn’t exist half-downloaded
            if fpath.is_file():
                fpath.unlink()
            raise
