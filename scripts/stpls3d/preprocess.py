from pathlib import Path
import shutil

from datasets.preprocessing.stpls3d_preprocessing import STPLS3DPreprocessing

# data_dir = Path("/workspace/host/home/Safe/OCL3D/POC2/transient_data/Mask3D_data")
data_dir = Path("data")
shutil.rmtree(data_dir / "processed/stpls3d", ignore_errors=True)

pp = STPLS3DPreprocessing(
    data_dir=str(data_dir / "raw/stpls3d"),
    save_dir=str(data_dir / "processed/stpls3d"),
)
pp.preprocess()
# pp.preprocess_sequential()
