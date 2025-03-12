import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from ai4bmr_core.utils.logging import get_logger
from pydantic_settings import BaseSettings

"""
NOTES:
- Should we rename base_dir to cache_dir?
"""


class BaseDatasetConfig(BaseSettings):
    id: str = None
    name: str = None
    description: str = None

    base_dir: None | Path = None
    cache_dir: None | Path = None
    raw_dir: None | Path = None
    processed_dir: None | Path = None

    urls: None | dict[str, str] = None
    raw_files: list[Path] = []
    processed_files: list[Path] = []


logger = get_logger("BaseDataset")


class BaseDataset(ABC):
    def __init__(
        self,
        *,
        name: str = None,
        id: str = None,
        description: str = None,
        base_dir: Path = None,
        raw_dir: Path = None,
        processed_dir: Path = None,
        urls: dict[str, str] = None,
        raw_files: list[Path] = None,
        processed_files: list[Path] = None,
        force_download: bool = False,
        force_process: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.id = id
        self.name = name
        self.description = description
        self.force_download = force_download
        self.force_process = force_process
        self.urls = urls
        self._is_setup = False

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
                Paths are over-configured. You are setting `base_dir` and `raw_dir` or `processed_dir` at the same time.
                base_dir (\033[93mignored\033[0m): {base_dir}
                raw_dir (\033[92mused\033[0m): {raw_dir}
                processed_dir (\033[92mused\033[0m): {processed_dir}
                """
            )

        logger.info("Initializing dataset... 🚀")

        self.base_dir = (
            base_dir
            if base_dir
            else (
                Path.home() / ".cache" / "ai4bmr" / "datasets" / self.name
                if self.name
                else None
            )
        )
        self.raw_dir = (
            raw_dir if raw_dir else self.base_dir / "01_raw" if self.base_dir else None
        )
        self.processed_dir = (
            processed_dir
            if processed_dir
            else self.base_dir / "02_processed" if self.base_dir else None
        )
        # NOTE: if no raw_files are given we use the urls names as fallback
        self.raw_files = (
            raw_files
            if raw_files
            else (
                [self.raw_dir / i for i in self.urls]
                if self.urls and self.raw_dir
                else []
            )
        )
        self.processed_files = processed_files or []

        logger.info(
            f"Dataset initialized 🎉 with \n\traw_dir file://{self.raw_dir}\n\tprocessed_dir file://{self.processed_dir}"
        )

    def setup(self):
        self._setup()
        return self

    def _setup(self):
        if self.force_download and self.raw_dir.exists():
            # NOTE: we delete the `raw_dir` if  `force_download` is True to ensure that all files are newly
            #  downloaded and not just some of them in case of an exception occurs during the download.
            #  Furthermore, we check if the folder exists and do not use `is_downloaded` as this is only True if the
            #  previously attempted downloads were successful.
            shutil.rmtree(self.raw_dir)

        if self.force_process and self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)

        if self.urls and (self.force_download or not self.is_downloaded):
            self.download()

        if not self.processed_files:
            logger.info("Processing... 🏃🏽")
        elif self.is_cached and not self.force_process:
            logger.info(
                "Cached data found 🥳. Loading data from cache. To re-process pass `force_process=True`."
            )
            self._data = self.load_cache()
        elif self.is_cached:
            logger.info(
                "Cached data found 🥳 but `force_process=True`. Processing... 🏃🏽"
            )
        else:
            logger.info("No cached data found 😔. Processing... 🏃🏽")

        self._data = self.load()
        if self.processed_dir:
            self.save_cache(self._data)

        self._is_setup = True

    def __getitem__(self, idx):
        if not self._is_setup:
            self.setup()
        return self._data[idx]

    def __len__(self):
        if not self._is_setup:
            self.setup()
        return len(self._data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

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
            for file_name, url in self.urls.items():
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
                with (
                    tqdm(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        unit_divisor=1024,
                        total=total if total is None else int(total),
                    ) as t,
                    fpath.open("wb") as f,
                ):
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

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: Path | str):
        self._base_dir = Path(value).expanduser() if value else None

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @raw_dir.setter
    def raw_dir(self, value: Path | str):
        self._raw_dir = Path(value).expanduser() if value else None

    @property
    def processed_dir(self) -> Path:
        return self._processed_dir

    @processed_dir.setter
    def processed_dir(self, value: Path | str):
        self._processed_dir = Path(value).expanduser() if value else None

    @property
    def raw_files(self) -> list[str]:
        return self._raw_files

    @raw_files.setter
    def raw_files(self, value: list[str]):
        self._raw_files = value if value else None

    @property
    def is_downloaded(self) -> bool:
        return (
            self.raw_dir
            and self.raw_files
            and all([(self.raw_dir / i).exists() for i in self.raw_files])
        )

    @property
    def processed_files(self) -> list[str]:
        return self._processed_files

    @processed_files.setter
    def processed_files(self, value: list[str]):
        self._processed_files = value

    @property
    def is_cached(self) -> bool:
        # NOTE: we need to check if processed_files is defined, otherwise caching is not enabled
        return (
            self.processed_dir
            and self.processed_files
            and all([(self.processed_dir / i).exists() for i in self.processed_files])
        )
