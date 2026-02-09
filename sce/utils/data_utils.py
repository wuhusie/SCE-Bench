import pandas as pd
import os
from pathlib import Path

def load_with_cache(file_path: Path, description: str, cache_dir: Path) -> pd.DataFrame:
    """
    General data loading function with caching mechanism to accelerate loading.
    
    Arguments:
        file_path (Path): Source file path (Path object).
        description (str): Data description for logging.
        cache_dir (Path): Cache directory path.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = f"{file_path.stem}.pkl"
    cache_path = cache_dir / cache_filename
    
    use_cache = False
    
    # 1. Check if cache exists and is not expired
    if cache_path.exists():
        try:
            # Compatibility with Windows/Linux modification time acquisition
            if file_path.exists():
                src_mtime = os.path.getmtime(file_path)
                cache_mtime = os.path.getmtime(cache_path)
                
                # Use cache if cache file is newer than source file
                if cache_mtime > src_mtime:
                    use_cache = True
                else:
                    print(f"🔄 Detected update in source file {file_path.name}, ignoring old cache.")
            else:
                # Source file doesn't exist (maybe only cache is needed), or path is incorrect.
                # Assuming if source is missing but cache exists, we can use cache for now.
                print(f"⚠️ Source file {file_path} not found, attempting to read from cache...")
                use_cache = True
        except OSError:
            pass
            
    if use_cache:
        print(f"🚀 [Fast Load] Reading {description} from cache ({cache_path.name})...")
        try:
            df = pd.read_pickle(cache_path)
            return df
        except Exception as e:
            print(f"⚠️ Cache read failed: {e}, falling back to source file.")

    # 2. Read source file (slow)
    if not file_path.exists():
         raise FileNotFoundError(f"❌ Source file not found: {file_path}")

    print(f"🐢 [First Load] Reading Excel source file {description} ({file_path.name}), this may take a while...")
    try:
        if file_path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to read source file: {e}")
    
    # 3. Save cache
    print(f"💾 Creating acceleration cache for {description} in {cache_dir.name} directory...")
    try:
        df.to_pickle(cache_path)
    except Exception as e:
        print(f"⚠️ Warning: Cache file save failed: {e}")
        
    return df

def load_macro_indicators(macro_dir: Path, cache_dir: Path):
    """
    Load macro-economic indicator data.
    """
    cpi_path = macro_dir / "CPI_Inflation_YoY_Full.xlsx"
    fedfunds_path = macro_dir / "FEDFUNDS.xlsx"
    unrate_path = macro_dir / "UNRATE.xlsx"

    print("Loading macro-economic indicator data...")
    return {
        "cpi": load_with_cache(cpi_path, "CPI Data", cache_dir),
        "interest": load_with_cache(fedfunds_path, "Interest Rate Data", cache_dir),
        "unemployment": load_with_cache(unrate_path, "Unemployment Rate Data", cache_dir)
    }
