from pathlib import Path

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai4bmr_core.data_models import MixIns


class Dataset(BaseSettings, MixIns.CreateFolderHierarchy):
    # dataset fields, required
    _id: str  # protected, this should not be set by the user
    _name: str  # protected, this should not be set by the user
    _data: None  # note: type defined on the subclass as we don't know it here yet
    _is_setup: bool = False # protected, this should not be set by the user

    # dataset fields, optional
    _description: str = ""
    _urls: None | dict[str, str] = None

    # note: the `dir` is resolved by self._base_dir ➡️ self.cache_dir ➡️ Path.home() / ".cache" / "ai4bmr"
    # dir: None | Path = None
    _base_dir: None | Path = None
    cache_dir: None | Path = None
    _raw_dir: None | Path = None
    _processed_dir: None | Path = None

    # user fields, optional
    force_download: bool = False
    force_process: bool = False

    @computed_field
    @property
    def base_dir(self) -> Path:
        if self._base_dir:
            return Path(self._base_dir)
        elif self.cache_dir:
            return Path(self.cache_dir) / self._name
        else:
            return Path.home() / ".cache" / "ai4bmr" / "datasets" / self._name

    @base_dir.setter
    def base_dir(self, value: Path | str):
        self._base_dir = Path(value) if value else None

    @computed_field
    @property
    def raw_dir(self) -> Path:
        return self._raw_dir if self._raw_dir else self.base_dir / "01_raw"

    @raw_dir.setter
    def raw_dir(self, value: Path | str):
        self._raw_dir = Path(value) if value else None

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return (
            self._processed_dir
            if self._processed_dir
            else self.base_dir / "02_processed"
        )

    @processed_dir.setter
    def processed_dir(self, value: Path | str):
        self._processed_dir = Path(value) if value else None

    @computed_field
    @property
    def raw_files(self) -> list[Path]:
        # TODO: this should not depend on the `self._urls` field but on a own attribute `self._raw_files`
        #   otherwise, we would always need to define the `download` method on the Dataset class as setting `_url`
        #   triggers download.
        return (
            [self.raw_dir / i for i in self._urls]
            if self.raw_dir and self._urls
            else []
        )

    @computed_field
    @property
    def is_downloaded(self) -> bool:
        # TODO: check `raw_files` todo.
        return all([i.exists() for i in self.raw_files]) if self.raw_files else False

    @computed_field
    @property
    def processed_files(self) -> list[Path]:
        # define a list of files that are produced by self.load of the dataset class
        return []

    @computed_field
    @property
    def is_cached(self) -> bool:
        return (
            all([i.exists() for i in self.processed_files])
            if self.processed_files
            else False
        )
