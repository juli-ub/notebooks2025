from pathlib import Path

class DataLoader:
    """Loads one text file from a directory."""

    def __init__(self, folder: str | Path, filename: str) -> None:
        """
        Parameters
        ----------
        folder : str | Path
            Folder that contains the text file.
        filename : str
            Name of the text file to load.
        """
        self.folder = Path(folder).expanduser().resolve()
        self.filename = filename
        self._content: str | None = None

    def load(self) -> None:
        """Reads the file into ``self._content``."""
        file_path = self.folder / self.filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            self._content = f.read()

    @property  
    def content(self) -> str | None:
        """Returns the loaded text (or ``None`` if ``load`` hasn't run)."""    
        return self._content

    def __repr__(self) -> str:
        return f"<DataLoader file={self.folder / self.filename}>"