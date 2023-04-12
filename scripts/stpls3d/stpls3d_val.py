import sys
import os

import main_instance_segmentation

os.environ["OMP_NUM_THREADS"]="3"

CURR_DBSCAN=14.0
CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=54

sys.argv.clear()
sys.argv.extend([
    f"main_instance_segmentation.py",
    f"general.experiment_name='validation'",
    f"general.project_name='stpls3d'",
    f"data/datasets=stpls3d",
    f"general.num_targets=15",
    f"data.num_labels=15",
    f"data.voxel_size=0.333",
    f"data.num_workers=8",
    f"data.cache_data=true",
    f"data.cropping_v1=false",
    f"general.reps_per_epoch=100",
    f"model.num_queries={CURR_QUERY}",
    f"general.on_crops=true",
    f"model.config.backbone._target_=models.Res16UNet18B",
    f"data.crop_length={CURR_SIZE}",
    f"general.eval_inner_core=50.0"
])

main_instance_segmentation.main()
