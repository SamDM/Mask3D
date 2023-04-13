from pathlib import Path
import shutil

from datasets.preprocessing.ocl3d_preprocessing import STPLS3DPreprocessing

inp_dir = Path("/workspace/host/root/media/robovision-syno5-work/nucleus/0039_OCL3D_data/PoC2/pipeline_py/mask3d")
out_dir = Path("data/processed")
rel_dir = Path("plant-nerf/nrot-003_ncam-003")

shutil.rmtree(out_dir / rel_dir, ignore_errors=True)

pp = STPLS3DPreprocessing(
    data_dir=str(inp_dir / rel_dir),
    save_dir=str(out_dir / rel_dir),
)
pp.preprocess()
# pp.preprocess_sequential()
